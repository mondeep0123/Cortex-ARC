"""
Object Cognition Primitive - FIXED SPATIAL VERSION

Learns to segment objects from background using PURE SEGMENTATION.
NO counting - that's Numerosity's job!

Task: Given a grid, predict binary mask: object (1) vs background (0)

Architecture: Fully Convolutional U-Net style to preserve spatial info
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from .base_primitive import BasePrimitiveModel, PrimitiveSkillConfig


class SpatialConvBlock(nn.Module):
    """Spatial-preserving convolutional block."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2  # Same padding
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


class ObjectCognitionPrimitive(BasePrimitiveModel):
    """
    Object Cognition - Fully Convolutional Segmentation Network.
    
    Uses U-Net style architecture to preserve spatial information
    for ANY grid size (3x3 to 30x30).
    
    Architecture:
    - Embedding: Color index → feature channels
    - Encoder: Convolutions with pooling (preserves spatial structure)
    - Decoder: Upsampling with skip connections
    - Output: Per-pixel segmentation mask
    """
    
    def __init__(self, config: PrimitiveSkillConfig):
        # Call parent but we won't use its encoder
        super().__init__(config)
        
        # Our U-Net replaces the encoder completely
        # Spatial feature dimensions
        # Start with fewer channels since we preserve spatial dims
        base_channels = 32
        
        # Color embedding (10 colors → base_channels)
        self.color_embedding = nn.Embedding(10, base_channels)
        
        # Encoder (down-sampling path with skip connections)
        self.enc1 = SpatialConvBlock(base_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)  # /2
        
        self.enc2 = SpatialConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)  # /4
        
        self.enc3 = SpatialConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)  # /8
        
        # Bottleneck
        self.bottleneck = SpatialConvBlock(256, 512)
        
        # Decoder (up-sampling path with skip connections)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = SpatialConvBlock(512, 256)  # 512 = 256 + 256 (skip)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = SpatialConvBlock(256, 128)  # 256 = 128 + 128 (skip)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = SpatialConvBlock(128, 64)  # 128 = 64 + 64 (skip)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Boundary head (auxiliary)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
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
        Forward pass - Fully Convolutional Segmentation.
        
        Args:
            grid_tensor: [batch, h, w] color indices
        
        Returns:
            Dictionary with:
            - 'segmentation': [batch, 1, h, w] object mask (0=bg, 1=object)
            - 'boundaries': [batch, 1, h, w] boundary detection
        """
        # Pad input to be divisible by 8 (for 3 pooling layers)
        grid_padded, original_size = self._pad_to_divisible(grid_tensor, divisor=8)
        orig_h, orig_w = original_size
        
        # Embed colors → spatial features [batch, h, w, channels]
        x = self.color_embedding(grid_padded.long())  # [batch, h, w, base_channels]
        x = x.permute(0, 3, 1, 2)  # [batch, channels, h, w]
        
        # Encoder with skip connections
        enc1 = self.enc1(x)  # [batch, 64, h, w]
        x = self.pool1(enc1)  # [batch, 64, h/2, w/2]
        
        enc2 = self.enc2(x)  # [batch, 128, h/2, w/2]
        x = self.pool2(enc2)  # [batch, 128, h/4, w/4]
        
        enc3 = self.enc3(x)  # [batch, 256, h/4, w/4]
        x = self.pool3(enc3)  # [batch, 256, h/8, w/8]
        
        # Bottleneck
        x = self.bottleneck(x)  # [batch, 512, h/8, w/8]
        
        # Decoder with skip connections
        x = self.up3(x)  # [batch, 256, h/4, w/4]
        x = torch.cat([x, enc3], dim=1)  # [batch, 512, h/4, w/4]
        x = self.dec3(x)  # [batch, 256, h/4, w/4]
        
        x = self.up2(x)  # [batch, 128, h/2, w/2]
        x = torch.cat([x, enc2], dim=1)  # [batch, 256, h/2, w/2]
        x = self.dec2(x)  # [batch, 128, h/2, w/2]
        
        x = self.up1(x)  # [batch, 64, h, w]
        x = torch.cat([x, enc1], dim=1)  # [batch, 128, h, w]
        x = self.dec1(x)  # [batch, 64, h, w]
        
        # Segmentation and boundary prediction
        segmentation = self.seg_head(x)  # [batch, 1, h_padded, w_padded]
        boundaries = self.boundary_head(x)  # [batch, 1, h_padded, w_padded]
        
        # Crop back to original size
        segmentation = segmentation[:, :, :orig_h, :orig_w]
        boundaries = boundaries[:, :, :orig_h, :orig_w]
        
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
        
        # Binary cross-entropy for segmentation
        seg_loss = F.binary_cross_entropy(seg_pred, seg_true)
        loss += 5.0 * seg_loss  # Heavy weight on segmentation!
        
        # 2. Boundary loss (AUXILIARY - 20% weight)
        if 'boundaries' in target and target['boundaries'] is not None:
            bound_pred = output['boundaries']
            bound_true = target['boundaries'].unsqueeze(1).float()
            
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
