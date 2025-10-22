"""
Wine Pouring Robot - Full Diffusion Policy Training Pipeline
Ready to push to HuggingFace!

Based on:
- Diffusion Policy (Chi et al., RSS 2023)
- RL-100 approach (IL → Offline RL → Online RL)
- Conditioning on overlap percentage and cup velocity

Dataset structure:
- Synthetic: linoyts/wan_pouring_liquid (vision pre-training)
- Real: Your robot demonstrations with overlap % annotations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple
import math


class SinusoidalPosEmb(nn.Module):
    """Positional embeddings for diffusion timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalDiffusionPolicy(nn.Module):
    """
    Diffusion Policy conditioned on:
    - overlap_pct: Circle overlap percentage (safety metric)
    - cup_velocity: Speed of cup movement
    - current_state: Current robot and cup positions
    """
    
    def __init__(
        self,
        state_dim: int = 4,  # [cup_x, cup_y, pour_x, pour_y]
        action_dim: int = 2,  # [delta_x, delta_y]
        condition_dim: int = 3,  # [overlap_pct, velocity_x, velocity_y]
        hidden_dim: int = 256,
        action_horizon: int = 8,
        num_diffusion_steps: int = 100
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_diffusion_steps = num_diffusion_steps
        
        # Time embedding
        time_dim = 64
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition encoder (overlap_pct, velocity)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Denoising network
        input_dim = action_horizon * action_dim
        self.denoiser = nn.Sequential(
            nn.Linear(input_dim + hidden_dim * 3, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Beta schedule for diffusion
        self.register_buffer('betas', self._cosine_beta_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
    
    def _cosine_beta_schedule(self, s=0.008):
        """Cosine schedule as proposed in improved DDPM"""
        steps = self.num_diffusion_steps
        x = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, noisy_actions, timestep, state, conditions):
        """
        Args:
            noisy_actions: [B, action_horizon, action_dim] - noisy action sequence
            timestep: [B] - diffusion timestep
            state: [B, state_dim] - current state
            conditions: [B, condition_dim] - [overlap_pct, velocity_x, velocity_y]
        
        Returns:
            predicted_noise: [B, action_horizon, action_dim]
        """
        # Encode time
        t_emb = self.time_mlp(timestep)
        
        # Encode state
        s_emb = self.state_encoder(state)
        
        # Encode conditions (overlap + velocity)
        c_emb = self.condition_encoder(conditions)
        
        # Flatten actions
        B = noisy_actions.shape[0]
        noisy_flat = noisy_actions.reshape(B, -1)
        
        # Concatenate all embeddings
        x = torch.cat([noisy_flat, t_emb, s_emb, c_emb], dim=-1)
        
        # Predict noise
        noise = self.denoiser(x)
        noise = noise.reshape(B, self.action_horizon, self.action_dim)
        
        return noise
    
    def add_noise(self, actions, timestep):
        """Add noise to actions for training"""
        noise = torch.randn_like(actions)
        
        # Get alpha values
        alpha_t = self.alphas_cumprod[timestep]
        alpha_t = alpha_t.reshape(-1, 1, 1)
        
        # Add noise: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        noisy_actions = torch.sqrt(alpha_t) * actions + torch.sqrt(1 - alpha_t) * noise
        
        return noisy_actions, noise
    
    @torch.no_grad()
    def sample(self, state, conditions, num_samples=1):
        """
        Sample action sequence using DDIM (faster than DDPM)
        
        Args:
            state: [B, state_dim]
            conditions: [B, condition_dim]
        
        Returns:
            actions: [B, action_horizon, action_dim]
        """
        B = state.shape[0]
        device = state.device
        
        # Start from random noise
        actions = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        
        # DDIM sampling with 10 steps (much faster than 100)
        timesteps = torch.linspace(self.num_diffusion_steps - 1, 0, 10, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(actions, t_batch, state, conditions)
            
            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            if i < len(timesteps) - 1:
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=device)
            
            # Predict x_0
            pred_x0 = (actions - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Update actions
            actions = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * predicted_noise
        
        return actions


class WinePouringDataset(Dataset):
    """
    Dataset for wine pouring demonstrations
    
    Each sample contains:
    - state: [cup_x, cup_y, pour_x, pour_y]
    - action: [action_horizon, 2] - trajectory of delta movements
    - overlap_pct: safety metric
    - cup_velocity: [vel_x, vel_y]
    """
    
    def __init__(self, demonstrations, action_horizon=8):
        self.demonstrations = demonstrations
        self.action_horizon = action_horizon
        self.samples = self._preprocess()
    
    def _preprocess(self):
        """Convert demonstrations into training samples"""
        samples = []
        
        for demo in self.demonstrations:
            # demo is a list of timesteps
            for i in range(len(demo) - self.action_horizon):
                state = demo[i]['state']  # [cup_x, cup_y, pour_x, pour_y]
                
                # Extract action chunk
                actions = []
                for j in range(i, i + self.action_horizon):
                    # Delta movement
                    delta = demo[j + 1]['pour_pos'] - demo[j]['pour_pos']
                    actions.append(delta)
                actions = np.array(actions)
                
                # Conditions
                overlap_pct = demo[i]['overlap'] / 100.0  # Normalize to [0, 1]
                velocity = demo[i]['cup_velocity']
                conditions = np.array([overlap_pct, velocity[0], velocity[1]])
                
                samples.append({
                    'state': state,
                    'actions': actions,
                    'conditions': conditions
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'state': torch.FloatTensor(sample['state']),
            'actions': torch.FloatTensor(sample['actions']),
            'conditions': torch.FloatTensor(sample['conditions'])
        }


def train_diffusion_policy(
    model: ConditionalDiffusionPolicy,
    train_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Training loop for diffusion policy"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            state = batch['state'].to(device)
            actions = batch['actions'].to(device)
            conditions = batch['conditions'].to(device)
            
            # Sample random timesteps
            B = actions.shape[0]
            timesteps = torch.randint(0, model.num_diffusion_steps, (B,), device=device)
            
            # Add noise to actions
            noisy_actions, noise = model.add_noise(actions, timesteps)
            
            # Predict noise
            predicted_noise = model.forward(noisy_actions, timesteps, state, conditions)
            
            # MSE loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return model


# Example usage and synthetic data generation
def generate_synthetic_demonstrations(num_demos=100, demo_length=80):
    """Generate synthetic demonstrations for testing"""
    demonstrations = []
    
    for _ in range(num_demos):
        demo = []
        
        # Random scenario
        cup_start = np.array([200.0, 200.0])
        pour_start = np.array([200.0, 200.0])
        
        scenario = np.random.choice(['smooth', 'drift', 'gust'])
        
        for t in range(demo_length):
            # Simulate cup movement
            if scenario == 'smooth':
                cup_pos = cup_start + np.array([t * 0.5, np.sin(t * 0.05) * 10])
            elif scenario == 'drift':
                cup_pos = cup_start + np.array([t * 1.5, t * 0.3])
            else:  # gust
                if t < 30:
                    cup_pos = cup_start
                else:
                    cup_pos = cup_start + np.array([(t - 30) * 5, (t - 30) * 3])
            
            # Simulate robot tracking
            pour_pos = pour_start + (cup_pos - cup_start) * 0.8  # Lag behind
            
            # Calculate overlap (simplified)
            distance = np.linalg.norm(cup_pos - pour_pos)
            overlap = max(0, 100 - distance * 2)
            
            # Calculate velocity
            if t > 0:
                cup_velocity = cup_pos - demo[-1]['cup_pos']
            else:
                cup_velocity = np.array([0.0, 0.0])
            
            state = np.concatenate([cup_pos, pour_pos])
            
            demo.append({
                'state': state,
                'cup_pos': cup_pos,
                'pour_pos': pour_pos,
                'overlap': overlap,
                'cup_velocity': cup_velocity
            })
        
        demonstrations.append(demo)
    
    return demonstrations


if __name__ == "__main__":
    print("="*60)
    print("Wine Pouring Robot - Diffusion Policy Training")
    print("="*60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic demonstrations...")
    demonstrations = generate_synthetic_demonstrations(num_demos=50)
    print(f"   Generated {len(demonstrations)} demonstrations")
    
    # Create dataset
    print("\n2. Creating dataset...")
    dataset = WinePouringDataset(demonstrations, action_horizon=8)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Initialize model
    print("\n3. Initializing diffusion policy...")
    model = ConditionalDiffusionPolicy(
        state_dim=4,
        action_dim=2,
        condition_dim=3,
        hidden_dim=256,
        action_horizon=8,
        num_diffusion_steps=100
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n4. Training model...")
    trained_model = train_diffusion_policy(
        model,
        train_loader,
        num_epochs=50,
        lr=3e-4
    )
    
    print("\n5. Testing sampling...")
    test_state = torch.FloatTensor([[200, 200, 200, 200]])
    test_conditions = torch.FloatTensor([[0.95, 0.5, 0.2]])  # high overlap, slow velocity
    sampled_actions = trained_model.sample(test_state, test_conditions)
    print(f"   Sampled action shape: {sampled_actions.shape}")
    
    print("\n6. Ready to save and push to HuggingFace!")
    print("   Next steps:")
    print("   - torch.save(trained_model.state_dict(), 'wine_robot_diffusion.pth')")
    print("   - Push to HuggingFace: huggingface-cli upload your-username/wine-robot-policy")
    print("   - Add model card with overlap % conditioning details")
    print("\n" + "="*60)