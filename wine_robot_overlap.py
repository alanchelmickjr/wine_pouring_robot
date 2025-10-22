import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

"""
Wine Pouring Robot with Diffusion Policy Conditioning

Inspired by RL-100 approach:
- Uses overlap percentage as conditioning signal
- Generates trajectory actions conditioned on safety state
- Can use Consistency Models for fast inference (wind gust reactions)
- Or action chunking for smooth coordinated pouring

Hugging Face models to consider:
- diffusion-policy/diffusion_policy_3d (Chi et al.)
- Columbia-AIR/consistency-policy (for fast single-step)
- Real-World RL datasets from robotics-diffusion-transformer
"""

class CircleOverlapCalculator:
    """Handles geometric calculations for circle overlap"""
    
    @staticmethod
    def calculate_overlap_percentage(pos1, pos2, r1, r2):
        """Calculate percentage overlap between two circles"""
        d = np.linalg.norm(pos1 - pos2)
        
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return 100.0
        
        # Intersection area calculation
        r1_sq = r1 * r1
        r2_sq = r2 * r2
        d_sq = d * d
        
        alpha = np.arccos((d_sq + r1_sq - r2_sq) / (2 * d * r1))
        beta = np.arccos((d_sq + r2_sq - r1_sq) / (2 * d * r2))
        
        intersection_area = (r1_sq * alpha + r2_sq * beta - 
                           0.5 * (r1_sq * np.sin(2 * alpha) + 
                                  r2_sq * np.sin(2 * beta)))
        
        cup_area = np.pi * r1_sq
        return (intersection_area / cup_area) * 100


class SimpleDiffusionPolicy:
    """
    Simplified diffusion policy for trajectory generation
    In production, replace with:
    - HuggingFace: diffusion_policy or consistency_policy
    - Conditioned on overlap_pct, cup_velocity, pour_state
    """
    
    def __init__(self, action_horizon=8):
        self.action_horizon = action_horizon
        
    def generate_trajectory(self, current_pos, target_pos, overlap_pct, 
                          cup_velocity, mode="normal"):
        """
        Generate action sequence conditioned on state
        
        Conditioning signals:
        - overlap_pct: Safety metric (0-100%)
        - cup_velocity: Speed of cup movement
        - mode: "emergency", "chase", "smooth"
        
        Returns: Array of positions (action chunk)
        """
        
        # In real implementation, this would be:
        # latents = torch.randn(action_horizon, 2)
        # for t in diffusion_timesteps:
        #     latents = model.denoise(latents, overlap_pct, cup_velocity, mode)
        # trajectory = model.decode(latents)
        
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return np.tile(current_pos, (self.action_horizon, 1))
        
        direction = direction / distance
        
        # Mode-based trajectory generation (simulating diffusion output)
        if mode == "emergency":
            # Pull back sharply
            trajectory = []
            for i in range(self.action_horizon):
                # Move away from target
                step_pos = current_pos - direction * (i + 1) * 5.0
                trajectory.append(step_pos)
            return np.array(trajectory)
        
        elif mode == "chase":
            # Aggressive tracking with increasing speed
            speed_multiplier = 2.0 + (100 - overlap_pct) / 50.0
            trajectory = []
            for i in range(self.action_horizon):
                progress = (i + 1) / self.action_horizon
                step_pos = current_pos + direction * distance * progress * speed_multiplier
                trajectory.append(step_pos)
            return np.array(trajectory)
        
        else:  # smooth
            # Smooth interpolation (like DDIM sampling)
            trajectory = []
            for i in range(self.action_horizon):
                progress = (i + 1) / self.action_horizon
                # Ease-in-out interpolation
                smooth_progress = progress * progress * (3 - 2 * progress)
                step_pos = current_pos + direction * distance * smooth_progress
                trajectory.append(step_pos)
            return np.array(trajectory)


class WinePouringRobotDiffusion:
    """
    Wine pouring robot using diffusion policy
    Similar to RL-100 approach but specialized for pouring
    """
    
    def __init__(self, cup_radius=50):
        self.cup_radius = cup_radius
        self.pour_circle_radius = cup_radius
        
        # State
        self.cup_pos = np.array([200.0, 200.0])
        self.pour_pos = np.array([200.0, 200.0])
        self.cup_velocity = np.array([0.0, 0.0])
        self.prev_cup_pos = np.array([200.0, 200.0])
        
        # Status
        self.is_pouring = True
        self.prev_overlap = 100.0
        
        # Policy
        self.policy = SimpleDiffusionPolicy(action_horizon=8)
        self.action_buffer = []
        self.action_idx = 0
        
        # Thresholds
        self.SAFE_OVERLAP = 60.0
        self.EMERGENCY_DROP = 40.0
        self.HIGH_VELOCITY_THRESHOLD = 10.0
        
        # Overlap calculator
        self.overlap_calc = CircleOverlapCalculator()
    
    def update_cup_velocity(self, new_cup_pos):
        """Calculate cup velocity for conditioning"""
        self.cup_velocity = new_cup_pos - self.prev_cup_pos
        self.prev_cup_pos = new_cup_pos.copy()
    
    def get_mode(self, overlap_pct, velocity_magnitude):
        """
        Determine policy mode based on state
        This is the key conditioning logic
        """
        overlap_drop = self.prev_overlap - overlap_pct
        
        # EMERGENCY: Sudden separation OR high velocity
        if overlap_drop > self.EMERGENCY_DROP or velocity_magnitude > self.HIGH_VELOCITY_THRESHOLD:
            return "emergency"
        
        # CHASE: Low overlap but tracking possible
        if overlap_pct < 80.0 and overlap_pct >= self.SAFE_OVERLAP:
            return "chase"
        
        # STOP: Can't recover
        if overlap_pct < self.SAFE_OVERLAP:
            return "emergency"  # Treat as emergency stop
        
        # SMOOTH: Normal operation
        return "smooth"
    
    def update(self, new_cup_pos):
        """Main update loop - uses action chunking from diffusion policy"""
        
        # Update cup velocity
        self.update_cup_velocity(new_cup_pos)
        self.cup_pos = new_cup_pos
        velocity_magnitude = np.linalg.norm(self.cup_velocity)
        
        # Calculate overlap
        overlap_pct = self.overlap_calc.calculate_overlap_percentage(
            self.cup_pos, self.pour_pos,
            self.cup_radius, self.pour_circle_radius
        )
        
        # Determine mode
        mode = self.get_mode(overlap_pct, velocity_magnitude)
        
        # Check if we need to stop pouring
        if mode == "emergency":
            self.is_pouring = False
        
        # Generate new trajectory if needed (or use buffered actions)
        if len(self.action_buffer) == 0 or self.action_idx >= len(self.action_buffer):
            # Generate new action chunk from diffusion policy
            trajectory = self.policy.generate_trajectory(
                self.pour_pos, 
                self.cup_pos,
                overlap_pct,
                velocity_magnitude,
                mode
            )
            self.action_buffer = trajectory
            self.action_idx = 0
        
        # Execute next action from chunk
        if self.action_idx < len(self.action_buffer):
            self.pour_pos = self.action_buffer[self.action_idx]
            self.action_idx += 1
        
        # Update state
        self.prev_overlap = overlap_pct
        
        return {
            'action': mode.upper(),
            'overlap': overlap_pct,
            'velocity': velocity_magnitude,
            'cup_pos': self.cup_pos.copy(),
            'pour_pos': self.pour_pos.copy()
        }


# Simulation Scenarios
def simulate_slow_drift():
    """Cup slowly drifts - robot should track smoothly"""
    robot = WinePouringRobotDiffusion()
    history = []
    
    print("=== SCENARIO 1: Slow Drift (Smooth Tracking) ===")
    for t in range(80):
        # Slow drift
        new_cup_pos = np.array([200.0 + t * 1.5, 200.0 + np.sin(t * 0.1) * 20])
        state = robot.update(new_cup_pos)
        history.append(state)
        
        if t % 10 == 0:
            print(f"t={t:3d}: overlap={state['overlap']:5.1f}% "
                  f"vel={state['velocity']:5.2f} mode={state['action']:10s} "
                  f"pouring={robot.is_pouring}")
    
    return history


def simulate_wind_gust():
    """Sudden wind gust - emergency stop"""
    robot = WinePouringRobotDiffusion()
    history = []
    
    print("\n=== SCENARIO 2: Wind Gust (Emergency Stop) ===")
    for t in range(50):
        if t < 20:
            new_cup_pos = np.array([200.0, 200.0])
        elif t == 20:
            # GUST! 
            new_cup_pos = np.array([350.0, 150.0])
        else:
            new_cup_pos = np.array([350.0 + (t-20) * 5, 150.0 - (t-20) * 3])
        
        state = robot.update(new_cup_pos)
        history.append(state)
        
        print(f"t={t:3d}: overlap={state['overlap']:5.1f}% "
              f"vel={state['velocity']:5.2f} mode={state['action']:10s} "
              f"pouring={robot.is_pouring}")
        
        if not robot.is_pouring:
            print(">>> EMERGENCY STOP TRIGGERED <<<")
            break
    
    return history


def simulate_gentle_adjustment():
    """Guest adjusts glass position - robot adapts"""
    robot = WinePouringRobotDiffusion()
    history = []
    
    print("\n=== SCENARIO 3: Gentle Adjustment (Adaptive Tracking) ===")
    for t in range(60):
        if t < 20:
            new_cup_pos = np.array([200.0, 200.0])
        elif t < 35:
            # Gentle nudge
            progress = (t - 20) / 15.0
            new_cup_pos = np.array([200.0 + progress * 60, 200.0 + progress * 30])
        else:
            # Hold new position
            new_cup_pos = np.array([260.0, 230.0])
        
        state = robot.update(new_cup_pos)
        history.append(state)
        
        if t % 5 == 0:
            print(f"t={t:3d}: overlap={state['overlap']:5.1f}% "
                  f"vel={state['velocity']:5.2f} mode={state['action']:10s} "
                  f"pouring={robot.is_pouring}")
    
    return history


# Visualization
def visualize_scenario(history, title):
    """Visualize robot behavior"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    cup_traj = np.array([s['cup_pos'] for s in history])
    pour_traj = np.array([s['pour_pos'] for s in history])
    overlaps = [s['overlap'] for s in history]
    velocities = [s['velocity'] for s in history]
    
    # Plot 1: Trajectories
    ax1.set_xlim(min(cup_traj[:, 0].min(), pour_traj[:, 0].min()) - 100,
                 max(cup_traj[:, 0].max(), pour_traj[:, 0].max()) + 100)
    ax1.set_ylim(min(cup_traj[:, 1].min(), pour_traj[:, 1].min()) - 100,
                 max(cup_traj[:, 1].max(), pour_traj[:, 1].max()) + 100)
    ax1.set_aspect('equal')
    ax1.set_title(f'{title}\nTrajectories')
    ax1.plot(cup_traj[:, 0], cup_traj[:, 1], 'b-', label='Cup', linewidth=2)
    ax1.plot(pour_traj[:, 0], pour_traj[:, 1], 'r--', label='Pour aim', linewidth=2)
    
    # Final positions
    cup_circle = Circle(history[-1]['cup_pos'], 50, color='blue', alpha=0.3)
    pour_circle = Circle(history[-1]['pour_pos'], 50, color='red', alpha=0.3)
    ax1.add_patch(cup_circle)
    ax1.add_patch(pour_circle)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overlap over time
    ax2.plot(overlaps, 'g-', linewidth=2)
    ax2.axhline(y=60, color='orange', linestyle='--', label='Safe threshold', linewidth=2)
    ax2.axhline(y=80, color='blue', linestyle='--', label='Chase threshold', alpha=0.5)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Overlap %')
    ax2.set_title('Circle Overlap Percentage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)
    
    # Plot 3: Velocity over time
    ax3.plot(velocities, 'purple', linewidth=2)
    ax3.axhline(y=10, color='red', linestyle='--', label='Emergency threshold', linewidth=2)
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Cup velocity (px/frame)')
    ax3.set_title('Cup Movement Speed')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mode over time
    modes = [s['action'] for s in history]
    mode_colors = {'SMOOTH': 'green', 'CHASE': 'orange', 'EMERGENCY': 'red'}
    mode_values = {'SMOOTH': 0, 'CHASE': 1, 'EMERGENCY': 2}
    mode_numeric = [mode_values.get(m, 0) for m in modes]
    
    ax4.scatter(range(len(modes)), mode_numeric, 
               c=[mode_colors.get(m, 'gray') for m in modes], s=50, alpha=0.6)
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Policy Mode')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['SMOOTH', 'CHASE', 'EMERGENCY'])
    ax4.set_title('Diffusion Policy Mode Selection')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()


# Run all scenarios
if __name__ == "__main__":
    history1 = simulate_slow_drift()
    history2 = simulate_wind_gust()
    history3 = simulate_gentle_adjustment()
    
    print("\n" + "="*60)
    print("DIFFUSION POLICY CONDITIONING SUMMARY")
    print("="*60)
    print("Conditioning Signals:")
    print("  1. Overlap percentage (0-100%)")
    print("  2. Cup velocity magnitude")
    print("  3. Overlap drop rate")
    print("\nPolicy Modes:")
    print("  - SMOOTH: Normal tracking (overlap > 80%)")
    print("  - CHASE: Aggressive tracking (60% < overlap < 80%)")
    print("  - EMERGENCY: Stop pouring (overlap < 60% OR velocity > 10px/frame)")
    print("\nNext Steps:")
    print("  1. Replace SimpleDiffusionPolicy with HuggingFace model")
    print("  2. Collect real demonstration data for IL")
    print("  3. Train with offline RL (add rewards for smoothness)")
    print("  4. Fine-tune with online RL in simulation")
    print("  5. Deploy with Consistency Model for fast inference")
    
    # Uncomment to visualize
    # visualize_scenario(history1, "Slow Drift")
    # visualize_scenario(history2, "Wind Gust")
    # visualize_scenario(history3, "Gentle Adjustment")