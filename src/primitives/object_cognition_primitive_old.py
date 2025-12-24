"""
Primitive 1: Object Cognition

Small neural network that learns to:
- Detect objects (connected components)
- Count objects
- Identify object properties (size, color, position)

Trained on diverse curriculum tasks, then tested for generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from .base_primitive import BasePrimitiveModel, PrimitiveSkillConfig
except ImportError:
    from src.primitives.base_primitive import BasePrimitiveModel, PrimitiveSkillConfig


class ObjectCognitionPrimitive(BasePrimitiveModel):
    """
    Small model that learns object-level reasoning.
    
    Tasks:
    1. Object detection (output: object masks)
    2. Object counting (output: count)
    3. Object properties (output: size, color, etc.)
    """
    
    def __init__(self, config: PrimitiveSkillConfig):
        super().__init__(config)
        
        # Object Cognition Task: SEGMENTATION ONLY
        # Detect WHERE objects are, not HOW MANY
        
        hidden = config.hidden_dim
        
        # 1. Object Segmentation Decoder (PRIMARY TASK)
        # Predicts binary mask: object vs background
        self.segmentation_decoder = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Unflatten(1, (hidden * 2 // 16, 4, 4)),
            nn.ConvTranspose2d(hidden * 2 // 16, 128, 4, 2, 1),  # → 8x8
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # → 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),  # → 32x32
            nn.Sigmoid()  # Binary: object (1) or background (0)
        )
        
        # 2. Boundary Detection (AUXILIARY)
        # Detects object edges
        self.boundary_detector = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Unflatten(1, (hidden // 16, 4, 4)),
            nn.ConvTranspose2d(hidden // 16, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()  # Boundary probability
        )
    
    def forward(self, grid_tensor):
        """
        Forward pass - Object Cognition ONLY
        
        Args:
            grid_tensor: [batch, h, w] color indices
        
        Returns:
            Dictionary with:
            - 'segmentation': [batch, 1, h, w] object mask (0=bg, 1=object)
            - 'boundaries': [batch, 1, h, w] boundary detection
        """
        # Encode
        features = self.encoder(grid_tensor)  # [batch, hidden]
        
        # Segment objects
        segmentation = self.segmentation_decoder(features)  # [batch, 1, H, W]
        
        # Detect boundaries
        boundaries = self.boundary_detector(features)  # [batch, 1, H, W]
        
        # Resize to input size
        batch, h, w = grid_tensor.shape
        segmentation = F.interpolate(segmentation, size=(h, w), mode='bilinear')
        boundaries = F.interpolate(boundaries, size=(h, w), mode='bilinear')
        
        return {
            'segmentation': segmentation,
            'boundaries': boundaries
        }
    
    def compute_loss(self, output, target):
        """
        Compute loss for Object Cognition (SEGMENTATION ONLY).
        
        Target format:
        {
            'segmentation': [batch, h, w] binary mask (0=bg, 1=object)
            'boundaries': [batch, h, w] boundary mask (optional)
        }
        """
        loss = 0.0
        
        # 1. Segmentation loss (PRIMARY - 80% weight)
        seg_pred = output['segmentation']  # [batch, 1, h, w]
        seg_true = target['segmentation'].unsqueeze(1).float()  # [batch, 1, h, w]
        
        # Resize if needed
        if seg_pred.shape != seg_true.shape:
            seg_true = F.interpolate(seg_true, size=seg_pred.shape[2:], mode='nearest')
        
        # Binary cross-entropy for segmentation
        seg_loss = F.binary_cross_entropy(seg_pred, seg_true)
        loss += 5.0 * seg_loss  # Heavy weight on segmentation!
        
        # 2. Boundary loss (AUXILIARY - 20% weight)
        if 'boundaries' in target and target['boundaries'] is not None:
            bound_pred = output['boundaries']
            bound_true = target['boundaries'].unsqueeze(1).float()
            
            if bound_pred.shape != bound_true.shape:
                bound_true = F.interpolate(bound_true, size=bound_pred.shape[2:], mode='nearest')
            
            bound_loss = F.binary_cross_entropy(bound_pred, bound_true)
            loss += 1.0 * bound_loss
        
        return loss
    
    def evaluate_output(self, output, target):
        """
        Evaluate accuracy using IoU (Intersection over Union).
        """
        seg_pred = output['segmentation'].squeeze()  # [batch, h, w]
        seg_true = target['segmentation'].float()  # [batch, h, w]
        
        # Binarize predictions
        seg_pred_binary = (seg_pred > 0.5).float()
        
        # Resize if needed
        if seg_pred_binary.shape != seg_true.shape:
            seg_true = F.interpolate(
                seg_true.unsqueeze(1), 
                size=seg_pred_binary.shape[1:], 
                mode='nearest'
            ).squeeze(1)
        
        # Compute IoU per sample
        intersection = (seg_pred_binary * seg_true).sum(dim=[1, 2])
        union = ((seg_pred_binary + seg_true) > 0).float().sum(dim=[1, 2])
        
        # Avoid division by zero
        iou = torch.where(
            union > 0,
            intersection / union,
            torch.ones_like(intersection)
        )
        
        # Return mean IoU as accuracy
        return iou.mean().item()
    
    def predict(self, grid: np.ndarray) -> dict:
        """
        High-level prediction interface.
        
        Args:
            grid: [h, w] numpy array
        
        Returns:
            dict with segmentation predictions
        """
        self.eval()
        with torch.no_grad():
            grid_tensor = torch.from_numpy(grid).long().unsqueeze(0)
            grid_tensor = grid_tensor.to(self.config.device)
            
            output = self(grid_tensor)
            
            # Get binary masks
            seg_mask = (output['segmentation'].squeeze().cpu().numpy() > 0.5)
            bound_mask = (output['boundaries'].squeeze().cpu().numpy() > 0.5)
            
            return {
                'object_mask': seg_mask.astype(float),
                'boundary_mask': bound_mask.astype(float),
                'segmentation_confidence': output['segmentation'].squeeze().cpu().numpy()
            }


# Example usage
if __name__ == "__main__":
    print("Object Cognition Primitive - Small Specialist Model")
    print("=" * 60)
    
    # Create small model
    config = PrimitiveSkillConfig(
        name="object_cognition",
        hidden_dim=64,  # Small!
        num_layers=2,
        learning_rate=1e-3
    )
    
    model = ObjectCognitionPrimitive(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel size: {total_params:,} parameters")
    print(f"Memory: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test forward pass
    test_grid = torch.randint(0, 10, (2, 10, 10))  # Batch of 2, 10x10 grids
    output = model(test_grid)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {test_grid.shape}")
    print(f"  Count output: {output['count'].shape}")
    print(f"  Mask output: {output['masks'].shape}")
    print(f"  Properties: {output['properties'].shape}")
    
    print(f"\n✓ Model ready for training!")
