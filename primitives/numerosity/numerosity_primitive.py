"""
Numerosity Primitive - Density Estimation Version

Task: Count objects, colors, and compare quantities

Architecture: U-Net + Density Map Regression (Crowd Counting approach)
- Uses U-Net to preserve spatial details
- Outputs a density map (one per color)
- Sums density map to get counts
- Handles variable sizes perfectly (Sum Pooling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

# Import from parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.primitives.base_primitive import BasePrimitiveModel, PrimitiveSkillConfig


class SpatialConvBlock(nn.Module):
    """Spatial-preserving convolutional block (same as Object Cognition)."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class NumerosityPrimitive(BasePrimitiveModel):
    """
    Numerosity - Density Estimation Network.
    
    Uses U-Net to predict a density map where sum(density) = count.
    This resolves the "global pooling" issue by forcing the model
    to localize objects before counting them.
    
    Outputs:
    - total_count: Sum of all color densities
    - color_counts: Sum of density per color channel
    - max_color: Derived from color counts
    """
    
    def __init__(self, config: PrimitiveSkillConfig):
        super().__init__(config)
        
        # U-Net Architecture (Same as Object Cognition)
        base_channels = 32
        self.color_embedding = nn.Embedding(10, base_channels)
        
        # Encoder
        self.enc1 = SpatialConvBlock(base_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = SpatialConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = SpatialConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = SpatialConvBlock(256, 512)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = SpatialConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = SpatialConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = SpatialConvBlock(128, 64)
        
        # Density Head (Predicts 10 channels, one per color)
        self.density_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 10, 1), # 10 color channels
            nn.ReLU() # Density must be non-negative
        )
        
    def _pad_to_divisible(self, x, divisor=8):
        """Pad grid to be divisible by divisor for pooling."""
        b, h, w = x.shape
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0)
        
        return x, (h, w)
    
    def forward(self, grid_tensor):
        """
        Forward pass - Density Estimation.
        """
        # Embed colors
        x = self.color_embedding(grid_tensor.long())
        x = x.permute(0, 3, 1, 2)
        
        # Pad
        grid_padded, original_size = self._pad_to_divisible(x, divisor=8)
        orig_h, orig_w = original_size
        x = grid_padded
        
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Density Prediction
        density = self.density_head(x) # [batch, 10, h_pad, w_pad]
        
        # Crop to original size
        density = density[:, :, :orig_h, :orig_w]
        
        # Sum Pooling to get Counts
        # Sum over spatial dims (H, W)
        color_counts = density.sum(dim=(2, 3)) # [batch, 10]
        
        # Total count is sum of color counts (excluding background if needed, but benchmark wants total non-bg)
        # Usually color 0 is background.
        # Benchmark "Total" means "non-background".
        # So sum(color_counts[:, 1:])
        non_bg_counts = color_counts[:, 1:]
        total_count = non_bg_counts.sum(dim=1, keepdim=True) # [batch, 1]
        
        # Use color_counts for max_color logits logic (largest count = likely max color)
        # We can just return color_counts and let loss function/eval handle argmax
        
        return {
            'total_count': total_count,
            'color_counts': color_counts,
            'max_color_logits': color_counts # Used for max color prediction
        }
    
    def compute_loss(self, output, target):
        """
        Compute loss for Numerosity (Regression).
        """
        loss = 0.0
        
        # 1. Total count loss (MSE)
        if 'total_count' in target:
            pred_total = output['total_count'].squeeze()
            true_total = target['total_count'].float()
            # Normalize for stability (counts 0-30)
            loss += 3.0 * F.mse_loss(pred_total / 30.0, true_total / 30.0)
        
        # 2. Per-color count loss (MSE)
        if 'color_counts' in target:
            pred_colors = output['color_counts']
            true_colors = target['color_counts'].float()
            loss += 2.0 * F.mse_loss(pred_colors / 10.0, true_colors / 10.0)
            
        # 3. Max color loss (Cross Entropy based on counts)
        if 'max_color' in target:
            # We treat color_counts as logits for max color classification
            # But they are positive counts.
            # Argmax(counts) should match true_max.
            # We can use CrossEntropy on "counts" interpreted as logits?
            # Or just rely on MSE to align counts.
            # Let's add explicit CE loss to encourage correct "winner"
            pred_logits = output['color_counts'].clone()
            # Mask background (index 0) from being max? Benchmark max_color is 0 if empty.
            # If target max_color is 0, then we want index 0 to be high.
            # This works.
            
            true_max = target['max_color'].long()
            loss += 0.5 * F.cross_entropy(pred_logits, true_max)
            
        return loss
    
    def evaluate_output(self, output, target):
        """
        Evaluate accuracy using count accuracy (within ±1).
        """
        pred_count = output['total_count'].squeeze().round()
        true_count = target['total_count'].float()
        
        accurate = (torch.abs(pred_count - true_count) <= 0.5).float() # Stricter for density?
        # Let's keep ±1 tolerance
        accurate = (torch.abs(pred_count - true_count) <= 1.0).float()
        
        return accurate.mean().item()
    
    def predict(self, grid: np.ndarray) -> dict:
        """
        High-level prediction interface.
        """
        self.eval()
        with torch.no_grad():
            grid_tensor = torch.from_numpy(grid).long().unsqueeze(0)
            grid_tensor = grid_tensor.to(self.config.device)
            
            output = self(grid_tensor)
            
            total = int(round(output['total_count'].item()))
            colors = output['color_counts'].squeeze().round().long().cpu().numpy()
            max_color = output['max_color_logits'].argmax(dim=1).item()
            
            return {
                'total_count': total,
                'color_counts': colors.tolist(),
                'max_color': max_color,
                'total_confidence': 1.0 # Placeholder
            }
