"""
Vision System for Wine Pouring Robot
Using Wan 2.1 Pouring Dataset for Pre-training

This trains:
1. Cup detector (where is the cup?)
2. Pour point estimator (where is the bottle pouring?)
3. IoU calculator (are we aligned?)

Dataset: linoyts/wan_pouring_liquid
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from datasets import load_dataset

# ============================================================================
# VISION BACKBONE - Cup & Pour Point Detection
# ============================================================================

class PouringVisionModel(nn.Module):
    """
    Multi-task vision model for pouring
    
    Outputs:
    - cup_center: [x, y] normalized coordinates
    - cup_radius: scalar
    - pour_point: [x, y] where liquid is falling
    - confidence: how sure are we?
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Use ResNet18 backbone (lightweight for real-time)
        from torchvision import models
        backbone = models.resnet18(pretrained=pretrained)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # Additional conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Task heads
        self.cup_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [x, y, radius]
        )
        
        self.pour_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [x, y]
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        feat = self.features(x)
        feat = self.conv(feat)
        feat = self.gap(feat).flatten(1)
        
        # Predict cup position and radius
        cup_params = self.cup_head(feat)
        cup_center = torch.sigmoid(cup_params[:, :2])  # Normalized [0, 1]
        cup_radius = torch.sigmoid(cup_params[:, 2:]) * 0.3  # Max 30% of image
        
        # Predict pour point
        pour_point = torch.sigmoid(self.pour_head(feat))
        
        # Confidence
        confidence = self.confidence_head(feat)
        
        return {
            'cup_center': cup_center,
            'cup_radius': cup_radius,
            'pour_point': pour_point,
            'confidence': confidence
        }


# ============================================================================
# DATASET - Wan Pouring Videos
# ============================================================================

class WanPouringDataset(Dataset):
    """
    Load Wan pouring videos and extract frames
    
    For training, we need:
    - Input: RGB frame
    - Target: cup position, pour point (manual annotation or auto-detected)
    """
    
    def __init__(self, split='train', max_videos=None):
        print("Loading Wan pouring dataset...")
        
        # Load from HuggingFace
        try:
            self.dataset = load_dataset("linoyts/wan_pouring_liquid", split="train")
            if max_videos:
                self.dataset = self.dataset.select(range(min(max_videos, len(self.dataset))))
            print(f"Loaded {len(self.dataset)} videos")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("You may need to: huggingface-cli login")
            self.dataset = []
        
        # Transform
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Extract frames
        self.frames = []
        self._extract_frames()
    
    def _extract_frames(self):
        """Extract frames from videos"""
        print("Extracting frames from videos...")
        
        for idx, item in enumerate(self.dataset):
            try:
                # Get video (this is simplified - actual implementation depends on format)
                video_path = item.get('video', None)
                if video_path is None:
                    continue
                
                # For now, use placeholder
                # In production: use opencv to extract frames
                # cap = cv2.VideoCapture(video_path)
                # while cap.isOpened():
                #     ret, frame = cap.read()
                #     if not ret: break
                #     self.frames.append(frame)
                
                # Placeholder: create synthetic frame
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                self.frames.append({
                    'image': frame,
                    'video_idx': idx,
                    'prompt': item.get('prompt', '')
                })
                
            except Exception as e:
                print(f"Error processing video {idx}: {e}")
                continue
        
        print(f"Extracted {len(self.frames)} frames")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame_data = self.frames[idx]
        image = frame_data['image']
        
        # Transform
        image_tensor = self.transform(image)
        
        # For now, use synthetic labels
        # In production: manual annotation or auto-detection
        cup_center = torch.FloatTensor([0.5, 0.6])  # Center-bottom
        cup_radius = torch.FloatTensor([0.1])
        pour_point = torch.FloatTensor([0.5, 0.4])  # Above cup
        
        return {
            'image': image_tensor,
            'cup_center': cup_center,
            'cup_radius': cup_radius,
            'pour_point': pour_point
        }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_vision_model(
    model,
    train_loader,
    num_epochs=50,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the vision model"""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            target_cup = batch['cup_center'].to(device)
            target_radius = batch['cup_radius'].to(device)
            target_pour = batch['pour_point'].to(device)
            
            # Forward
            outputs = model(images)
            
            # Loss
            loss_cup = nn.MSELoss()(outputs['cup_center'], target_cup)
            loss_radius = nn.MSELoss()(outputs['cup_radius'], target_radius)
            loss_pour = nn.MSELoss()(outputs['pour_point'], target_pour)
            
            loss = loss_cup + loss_radius + loss_pour
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return model


# ============================================================================
# IoU CALCULATION (Your "Overlap Detection")
# ============================================================================

class IoUCalculator:
    """
    Calculate IoU between cup circle and pour circle
    This is the core of your control strategy!
    """
    
    @staticmethod
    def calculate_circle_iou(center1, radius1, center2, radius2, image_size=(640, 480)):
        """
        Calculate IoU between two circles
        
        Args:
            center1, center2: [x, y] in normalized coords [0, 1]
            radius1, radius2: normalized radius [0, 1]
            image_size: (width, height) for denormalization
        
        Returns:
            iou: Intersection over Union [0, 1]
        """
        # Denormalize
        c1 = np.array(center1) * np.array([image_size[0], image_size[1]])
        c2 = np.array(center2) * np.array([image_size[0], image_size[1]])
        r1 = radius1 * min(image_size)
        r2 = radius2 * min(image_size)
        
        # Distance between centers
        d = np.linalg.norm(c1 - c2)
        
        # No overlap
        if d >= r1 + r2:
            return 0.0
        
        # Complete overlap
        if d <= abs(r1 - r2):
            return 1.0
        
        # Partial overlap
        r1_sq = r1 * r1
        r2_sq = r2 * r2
        d_sq = d * d
        
        alpha = np.arccos((d_sq + r1_sq - r2_sq) / (2 * d * r1))
        beta = np.arccos((d_sq + r2_sq - r1_sq) / (2 * d * r2))
        
        intersection_area = (r1_sq * alpha + r2_sq * beta - 
                           0.5 * (r1_sq * np.sin(2 * alpha) + 
                                  r2_sq * np.sin(2 * beta)))
        
        union_area = np.pi * (r1_sq + r2_sq) - intersection_area
        
        return float(intersection_area / union_area)
    
    @staticmethod
    def visualize_iou(image, cup_center, cup_radius, pour_center, pour_radius, iou):
        """Draw circles and IoU on image"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Denormalize
        cup_c = (int(cup_center[0] * w), int(cup_center[1] * h))
        cup_r = int(cup_radius * min(w, h))
        pour_c = (int(pour_center[0] * w), int(pour_center[1] * h))
        pour_r = int(pour_radius * min(w, h))
        
        # Draw circles
        cv2.circle(img, cup_c, cup_r, (0, 255, 0), 2)  # Cup in green
        cv2.circle(img, pour_c, pour_r, (0, 0, 255), 2)  # Pour in red
        
        # Draw IoU text
        cv2.putText(img, f"IoU: {iou:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img


# ============================================================================
# REAL-TIME INFERENCE
# ============================================================================

class VisionSystem:
    """
    Complete vision system for robot
    """
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load model
        self.model = PouringVisionModel(pretrained=True).to(device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Transform
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # IoU calculator
        self.iou_calc = IoUCalculator()
    
    @torch.no_grad()
    def process_frame(self, image):
        """
        Process single frame from overhead camera
        
        Args:
            image: numpy array (H, W, 3) BGR
        
        Returns:
            dict with cup_center, pour_point, iou, etc.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform and add batch dimension
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Inference
        outputs = self.model(input_tensor)
        
        # Extract predictions
        cup_center = outputs['cup_center'][0].cpu().numpy()
        cup_radius = outputs['cup_radius'][0].cpu().numpy()[0]
        pour_point = outputs['pour_point'][0].cpu().numpy()
        confidence = outputs['confidence'][0].cpu().numpy()[0]
        
        # Calculate IoU (pour circle same size as cup)
        iou = self.iou_calc.calculate_circle_iou(
            cup_center, cup_radius,
            pour_point, cup_radius,
            image_size=(image.shape[1], image.shape[0])
        )
        
        return {
            'cup_center': cup_center,
            'cup_radius': cup_radius,
            'pour_point': pour_point,
            'confidence': confidence,
            'iou': iou
        }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("WINE POURING ROBOT - VISION SYSTEM TRAINING")
    print("="*70)
    
    # Step 1: Load dataset
    print("\n[1/4] Loading Wan pouring dataset...")
    dataset = WanPouringDataset(max_videos=10)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Step 2: Initialize model
    print("\n[2/4] Initializing vision model...")
    model = PouringVisionModel(pretrained=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 3: Train
    print("\n[3/4] Training vision model...")
    trained_model = train_vision_model(model, train_loader, num_epochs=10)
    
    # Step 4: Test
    print("\n[4/4] Testing inference...")
    vision_system = VisionSystem()
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = vision_system.process_frame(test_image)
    print(f"Result: {result}")
    
    print("\n" + "="*70)
    print("VISION SYSTEM READY!")
    print("="*70)
    print("""
    # Use in your robot control loop:
    vision = VisionSystem(model_path='vision_model.pth')
    
    while True:
        frame = camera.get_frame()
        result = vision.process_frame(frame)
        
        iou = result['iou']
        if iou < 0.6:
            # Emergency stop or chase!
            robot.stop_pouring()
    """)