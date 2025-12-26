"""
Train Color-Aware Object Cognition Primitive (v2)

Enhanced Object Cognition with comprehensive color masking capabilities.
Based on the original ObjectCognitionPrimitive architecture (U-Net).

TASK TYPES (16 total):
1.  mask_all           - Mask all foreground
2.  mask_color_X       - Mask only color X (0-9)
3.  mask_except_X      - Mask all except color X  
4.  mask_dominant      - Mask most frequent color
5.  mask_rare          - Mask least frequent color
6.  mask_second_most   - Mask 2nd most frequent color
7.  mask_largest_region - Mask largest connected component
8.  mask_smallest_region - Mask smallest connected component
9.  mask_boundary      - Mask pixels on grid edge
10. mask_interior      - Mask non-edge pixels
11. mask_isolated      - Mask single isolated pixels
12. mask_connected_component_of_color - Mask connected region of specific color

IMPORTANT: Saves to SEPARATE checkpoint:
    checkpoints/color_aware_object_cognition_best.pt
    
Does NOT overwrite object_cognition_best.pt!

Author: Cortex-ARC Team
Date: December 26, 2025
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from dataclasses import dataclass
from scipy import ndimage


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass  
class ColorAwareConfig:
    """Configuration for Color-Aware Object Cognition."""
    name: str = "color_aware_object_cognition"
    base_channels: int = 32
    color_embed_dim: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    device: str = 'cuda'
    

# =============================================================================
# TASK TYPES ENUMERATION
# =============================================================================

class TaskType:
    """All supported masking task types."""
    
    # Basic masking (0-10)
    MASK_ALL = 0              # Mask all foreground
    MASK_COLOR_1 = 1          # Mask color 1
    MASK_COLOR_2 = 2          # Mask color 2
    MASK_COLOR_3 = 3          # Mask color 3
    MASK_COLOR_4 = 4          # Mask color 4
    MASK_COLOR_5 = 5          # Mask color 5
    MASK_COLOR_6 = 6          # Mask color 6
    MASK_COLOR_7 = 7          # Mask color 7
    MASK_COLOR_8 = 8          # Mask color 8
    MASK_COLOR_9 = 9          # Mask color 9
    
    # Frequency-based (10-12)
    MASK_DOMINANT = 10        # Most frequent color
    MASK_RARE = 11            # Least frequent color  
    MASK_SECOND_MOST = 12     # 2nd most frequent
    
    # Spatial (13-16)
    MASK_LARGEST_REGION = 13  # Largest connected component
    MASK_SMALLEST_REGION = 14 # Smallest connected component
    MASK_BOUNDARY = 15        # Edge pixels
    MASK_INTERIOR = 16        # Non-edge pixels
    
    # Special (17-19)
    MASK_ISOLATED = 17        # Single pixels with no neighbors
    MASK_EXCEPT_DOMINANT = 18 # All except most frequent
    MASK_EXCEPT_RARE = 19     # All except least frequent
    
    NUM_TASK_TYPES = 20
    
    @classmethod
    def name(cls, task_id: int) -> str:
        """Get human-readable name for task type."""
        names = {
            0: "mask_all", 
            1: "mask_color_1", 2: "mask_color_2", 3: "mask_color_3",
            4: "mask_color_4", 5: "mask_color_5", 6: "mask_color_6",
            7: "mask_color_7", 8: "mask_color_8", 9: "mask_color_9",
            10: "mask_dominant", 11: "mask_rare", 12: "mask_second_most",
            13: "mask_largest_region", 14: "mask_smallest_region",
            15: "mask_boundary", 16: "mask_interior",
            17: "mask_isolated", 18: "mask_except_dominant", 19: "mask_except_rare"
        }
        return names.get(task_id, f"unknown_{task_id}")


# =============================================================================
# MODEL: Color-Aware U-Net (based on original ObjectCognitionPrimitive)
# =============================================================================

class SpatialConvBlock(nn.Module):
    """Spatial-preserving convolutional block (from original)."""
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


class ColorAwareObjectCognition(nn.Module):
    """
    Color-Aware Object Cognition - U-Net with Task Conditioning.
    
    Based on the original ObjectCognitionPrimitive U-Net architecture.
    Adds task type conditioning to support color-specific masking.
    
    Architecture:
    - Color embedding: Grid values → feature channels
    - Task embedding: Task type → conditioning vector
    - U-Net encoder/decoder with skip connections
    - Task conditioning injected at bottleneck
    """
    
    def __init__(self, config: ColorAwareConfig):
        super().__init__()
        self.config = config
        base_channels = config.base_channels
        
        # Color embedding (10 colors → base_channels)
        self.color_embedding = nn.Embedding(10, base_channels)
        
        # Task type embedding (20 task types → embed_dim)
        self.task_embedding = nn.Embedding(TaskType.NUM_TASK_TYPES, config.color_embed_dim)
        
        # U-Net Encoder (from original)
        self.enc1 = SpatialConvBlock(base_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = SpatialConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = SpatialConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = SpatialConvBlock(256, 512)
        
        # Task conditioning projection
        self.task_projection = nn.Sequential(
            nn.Linear(config.color_embed_dim, 512),
            nn.ReLU(inplace=True)
        )
        
        # U-Net Decoder (from original)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = SpatialConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = SpatialConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = SpatialConvBlock(128, 64)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Optimizer and scheduler (initialized during training)
        self.optimizer = None
        self.scheduler = None
        self.best_val_iou = 0.0
    
    def _pad_to_divisible(self, x, divisor=8):
        """Pad grid to be divisible by divisor for pooling."""
        if x.dim() == 2:
            h, w = x.shape
        else:
            b, h, w = x.shape
        
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0)
        
        return x, (h, w)
    
    def forward(self, grid: torch.Tensor, task_type: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with task conditioning.
        
        Args:
            grid: [batch, h, w] color indices (0-9)
            task_type: [batch] task type indices
        
        Returns:
            mask: [batch, 1, h, w] segmentation mask probabilities
        """
        # Pad input
        grid_padded, original_size = self._pad_to_divisible(grid, divisor=8)
        orig_h, orig_w = original_size
        
        # Embed colors
        x = self.color_embedding(grid_padded.long())  # [batch, h, w, base_channels]
        x = x.permute(0, 3, 1, 2)  # [batch, channels, h, w]
        
        # Encoder with skip connections
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Task conditioning: add task embedding to bottleneck
        task_emb = self.task_embedding(task_type)  # [batch, embed_dim]
        task_proj = self.task_projection(task_emb)  # [batch, 512]
        task_cond = task_proj.unsqueeze(-1).unsqueeze(-1)  # [batch, 512, 1, 1]
        x = x + task_cond  # Broadcast addition
        
        # Decoder with skip connections
        x = self.up3(x)
        # Handle size mismatch
        if x.shape[2:] != enc3.shape[2:]:
            x = F.interpolate(x, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        if x.shape[2:] != enc2.shape[2:]:
            x = F.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        if x.shape[2:] != enc1.shape[2:]:
            x = F.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Segmentation
        seg = self.seg_head(x)
        
        # Crop back to original size
        seg = seg[:, :, :orig_h, :orig_w]
        
        return seg
    
    def setup_training(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs,
            eta_min=1e-6
        )
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config,
            'best_val_iou': self.best_val_iou
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint.get('optimizer_state_dict') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_iou = checkpoint.get('best_val_iou', 0.0)


# =============================================================================
# CURRICULUM: Task Generator
# =============================================================================

def generate_mask(grid: np.ndarray, task_type: int) -> np.ndarray:
    """
    Generate ground truth mask for a given task type.
    
    Args:
        grid: [H, W] numpy array with color values 0-9
        task_type: Task type ID from TaskType enum
    
    Returns:
        mask: [H, W] binary mask (0 or 1)
    """
    H, W = grid.shape
    
    # Basic color masking
    if task_type == TaskType.MASK_ALL:
        return (grid > 0).astype(np.float32)
    
    elif 1 <= task_type <= 9:
        # Mask specific color
        return (grid == task_type).astype(np.float32)
    
    elif task_type == TaskType.MASK_DOMINANT:
        fg = grid[grid > 0]
        if len(fg) == 0:
            return np.zeros((H, W), dtype=np.float32)
        colors, counts = np.unique(fg, return_counts=True)
        dominant = colors[np.argmax(counts)]
        return (grid == dominant).astype(np.float32)
    
    elif task_type == TaskType.MASK_RARE:
        fg = grid[grid > 0]
        if len(fg) == 0:
            return np.zeros((H, W), dtype=np.float32)
        colors, counts = np.unique(fg, return_counts=True)
        rare = colors[np.argmin(counts)]
        return (grid == rare).astype(np.float32)
    
    elif task_type == TaskType.MASK_SECOND_MOST:
        fg = grid[grid > 0]
        if len(fg) == 0:
            return np.zeros((H, W), dtype=np.float32)
        colors, counts = np.unique(fg, return_counts=True)
        if len(colors) < 2:
            return np.zeros((H, W), dtype=np.float32)
        sorted_indices = np.argsort(counts)[::-1]
        second_most = colors[sorted_indices[1]]
        return (grid == second_most).astype(np.float32)
    
    elif task_type == TaskType.MASK_LARGEST_REGION:
        fg = (grid > 0).astype(np.int32)
        labeled, num_features = ndimage.label(fg)
        if num_features == 0:
            return np.zeros((H, W), dtype=np.float32)
        sizes = ndimage.sum(fg, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        return (labeled == largest_label).astype(np.float32)
    
    elif task_type == TaskType.MASK_SMALLEST_REGION:
        fg = (grid > 0).astype(np.int32)
        labeled, num_features = ndimage.label(fg)
        if num_features == 0:
            return np.zeros((H, W), dtype=np.float32)
        sizes = ndimage.sum(fg, labeled, range(1, num_features + 1))
        smallest_label = np.argmin(sizes) + 1
        return (labeled == smallest_label).astype(np.float32)
    
    elif task_type == TaskType.MASK_BOUNDARY:
        # Pixels on the edge of the grid
        mask = np.zeros((H, W), dtype=np.float32)
        mask[0, :] = (grid[0, :] > 0)
        mask[-1, :] = (grid[-1, :] > 0)
        mask[:, 0] = (grid[:, 0] > 0)
        mask[:, -1] = (grid[:, -1] > 0)
        return mask
    
    elif task_type == TaskType.MASK_INTERIOR:
        # Non-edge foreground pixels
        fg = (grid > 0).astype(np.float32)
        boundary = np.zeros_like(fg)
        boundary[0, :] = 1
        boundary[-1, :] = 1
        boundary[:, 0] = 1
        boundary[:, -1] = 1
        return fg * (1 - boundary)
    
    elif task_type == TaskType.MASK_ISOLATED:
        # Single pixels with no 4-connected neighbors
        fg = (grid > 0).astype(np.int32)
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        neighbor_count = ndimage.convolve(fg, kernel, mode='constant', cval=0)
        isolated = fg * (neighbor_count == 0)
        return isolated.astype(np.float32)
    
    elif task_type == TaskType.MASK_EXCEPT_DOMINANT:
        fg = grid[grid > 0]
        if len(fg) == 0:
            return np.zeros((H, W), dtype=np.float32)
        colors, counts = np.unique(fg, return_counts=True)
        dominant = colors[np.argmax(counts)]
        return ((grid > 0) & (grid != dominant)).astype(np.float32)
    
    elif task_type == TaskType.MASK_EXCEPT_RARE:
        fg = grid[grid > 0]
        if len(fg) == 0:
            return np.zeros((H, W), dtype=np.float32)
        colors, counts = np.unique(fg, return_counts=True)
        rare = colors[np.argmin(counts)]
        return ((grid > 0) & (grid != rare)).astype(np.float32)
    
    else:
        # Default: mask all foreground
        return (grid > 0).astype(np.float32)


class ColorAwareCurriculumDataset(Dataset):
    """
    Generates diverse curriculum tasks for Color-Aware Object Cognition.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        grid_size_range: Tuple[int, int] = (8, 20),  # Larger min size for U-Net
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.grid_size_range = grid_size_range
        self.rng = np.random.RandomState(seed)
        
        # Pre-generate samples
        self.samples = []
        print(f"Generating {num_samples} curriculum samples...")
        for i in range(num_samples):
            self.samples.append(self._generate_sample())
            if (i + 1) % 2500 == 0:
                print(f"  Generated {i+1}/{num_samples}")
    
    def _generate_sample(self) -> Tuple[np.ndarray, int, np.ndarray]:
        """Generate a single training sample."""
        
        # Random grid size (must be >= 8 for U-Net pooling)
        H = self.rng.randint(self.grid_size_range[0], self.grid_size_range[1] + 1)
        W = self.rng.randint(self.grid_size_range[0], self.grid_size_range[1] + 1)
        
        # Random number of colors (1-5)
        num_colors = self.rng.randint(1, 6)
        colors = self.rng.choice(range(1, 10), size=num_colors, replace=False)
        
        # Create grid with random objects/regions
        grid = np.zeros((H, W), dtype=np.int64)
        
        # Add random objects for each color
        for color in colors:
            # Random strategy: scattered pixels or blob regions
            strategy = self.rng.choice(['scatter', 'blob', 'line'])
            
            if strategy == 'scatter':
                num_pixels = self.rng.randint(1, max(2, H * W // 6))
                positions = self.rng.choice(H * W, size=min(num_pixels, H * W // 2), replace=False)
                for pos in positions:
                    r, c = pos // W, pos % W
                    if grid[r, c] == 0:
                        grid[r, c] = color
            
            elif strategy == 'blob':
                # Create a random blob using morphological dilation
                start_r = self.rng.randint(0, H)
                start_c = self.rng.randint(0, W)
                blob = np.zeros((H, W), dtype=np.int32)
                blob[start_r, start_c] = 1
                # Dilate randomly
                for _ in range(self.rng.randint(1, 4)):
                    blob = ndimage.binary_dilation(blob).astype(np.int32)
                # Apply to grid where empty
                grid = np.where((blob > 0) & (grid == 0), color, grid)
            
            elif strategy == 'line':
                # Horizontal or vertical line
                if self.rng.random() > 0.5:
                    # Horizontal
                    row = self.rng.randint(0, H)
                    c_start = self.rng.randint(0, W)
                    c_end = min(c_start + self.rng.randint(2, W // 2), W)
                    grid[row, c_start:c_end] = np.where(grid[row, c_start:c_end] == 0, color, grid[row, c_start:c_end])
                else:
                    # Vertical
                    col = self.rng.randint(0, W)
                    r_start = self.rng.randint(0, H)
                    r_end = min(r_start + self.rng.randint(2, H // 2), H)
                    grid[r_start:r_end, col] = np.where(grid[r_start:r_end, col] == 0, color, grid[r_start:r_end, col])
        
        # Choose task type with weighted distribution
        # More weight on common operations
        weights = np.array([
            0.15,  # MASK_ALL (common)
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,  # MASK_COLOR 1-9
            0.08,  # MASK_DOMINANT (useful)
            0.06,  # MASK_RARE (useful)
            0.04,  # MASK_SECOND_MOST
            0.05,  # MASK_LARGEST_REGION
            0.05,  # MASK_SMALLEST_REGION
            0.03,  # MASK_BOUNDARY
            0.03,  # MASK_INTERIOR
            0.03,  # MASK_ISOLATED
            0.04,  # MASK_EXCEPT_DOMINANT
            0.04,  # MASK_EXCEPT_RARE
        ])
        weights = weights / weights.sum()  # Normalize
        
        task_type = self.rng.choice(TaskType.NUM_TASK_TYPES, p=weights)
        
        # Ensure task is valid for this grid
        # If masking a color that doesn't exist, fall back to MASK_ALL
        if 1 <= task_type <= 9:
            if task_type not in grid:
                task_type = TaskType.MASK_ALL
        
        # Generate mask
        mask = generate_mask(grid, task_type)
        
        return grid, task_type, mask
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid, task_type, mask = self.samples[idx]
        return (
            torch.tensor(grid, dtype=torch.long),
            torch.tensor(task_type, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32)
        )


def collate_fn(batch):
    """Custom collate with padding for variable-sized grids."""
    grids, tasks, masks = zip(*batch)
    
    # Find max dimensions
    max_h = max(g.shape[0] for g in grids)
    max_w = max(g.shape[1] for g in grids)
    
    # Round up to multiple of 8 (for U-Net pooling)
    max_h = ((max_h + 7) // 8) * 8
    max_w = ((max_w + 7) // 8) * 8
    
    padded_grids = []
    padded_masks = []
    
    for g, m in zip(grids, masks):
        h, w = g.shape
        pad_h, pad_w = max_h - h, max_w - w
        
        padded_g = F.pad(g, (0, pad_w, 0, pad_h), value=0)
        padded_m = F.pad(m, (0, pad_w, 0, pad_h), value=0)
        
        padded_grids.append(padded_g)
        padded_masks.append(padded_m)
    
    return (
        torch.stack(padded_grids),
        torch.stack(list(tasks)),
        torch.stack(padded_masks).unsqueeze(1)
    )


def create_curriculum_loaders(
    train_size: int = 10000,
    val_size: int = 2000,
    test_size: int = 2000,
    batch_size: int = 32,
    grid_size_range: Tuple[int, int] = (8, 20)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders with different seeds."""
    
    print("\n" + "="*70)
    print("GENERATING CURRICULUM")
    print("="*70)
    
    train_dataset = ColorAwareCurriculumDataset(
        num_samples=train_size, grid_size_range=grid_size_range, seed=42
    )
    val_dataset = ColorAwareCurriculumDataset(
        num_samples=val_size, grid_size_range=grid_size_range, seed=43
    )
    test_dataset = ColorAwareCurriculumDataset(
        num_samples=test_size, grid_size_range=grid_size_range, seed=44
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader


# =============================================================================
# TRAINING
# =============================================================================

def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute mean IoU."""
    pred_binary = (pred > threshold).float()
    
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0
    
    return (intersection / union).item()


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    
    for grids, tasks, masks in loader:
        grids = grids.to(device)
        tasks = tasks.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        preds = model(grids, tasks)
        loss = criterion(preds, masks)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_iou += compute_iou(preds, masks)
        num_batches += 1
    
    return total_loss / num_batches, total_iou / num_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    
    for grids, tasks, masks in loader:
        grids = grids.to(device)
        tasks = tasks.to(device)
        masks = masks.to(device)
        
        preds = model(grids, tasks)
        loss = criterion(preds, masks)
        
        total_loss += loss.item()
        total_iou += compute_iou(preds, masks)
        num_batches += 1
    
    return total_loss / num_batches, total_iou / num_batches


def train_color_aware_object_cognition():
    """Main training function."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  COLOR-AWARE OBJECT COGNITION TRAINING  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    config = ColorAwareConfig(device=device)
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Base channels: {config.base_channels}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Task types: {TaskType.NUM_TASK_TYPES}")
    
    # Create model
    model = ColorAwareObjectCognition(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Model size: ~{params * 4 / 1024 / 1024:.2f} MB")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_curriculum_loaders(
        train_size=10000,
        val_size=2000,
        test_size=2000,
        batch_size=config.batch_size
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val:   {len(val_loader.dataset)}")
    print(f"  Test:  {len(test_loader.dataset)}")
    
    # Training setup
    model.setup_training()
    criterion = nn.BCELoss()
    
    checkpoint_path = "checkpoints/color_aware_object_cognition_best.pt"
    Path(checkpoint_path).parent.mkdir(exist_ok=True)
    
    print(f"\nCheckpoint: {checkpoint_path}")
    print("(Does NOT overwrite object_cognition_best.pt)")
    
    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    print(f"{'Epoch':<6} {'Train IoU':<12} {'Val IoU':<12} {'Status'}")
    print("-" * 50)
    
    best_val_iou = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(config.max_epochs):
        train_loss, train_iou = train_epoch(model, train_loader, model.optimizer, criterion, device)
        val_loss, val_iou = evaluate(model, val_loader, criterion, device)
        
        model.scheduler.step()
        
        status = ""
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            patience_counter = 0
            model.best_val_iou = best_val_iou
            model.save_checkpoint(checkpoint_path)
            status = "✓ BEST"
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or status:
            print(f"{epoch:<6} {train_iou:<12.4f} {val_iou:<12.4f} {status}")
        
        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        if val_iou >= 0.9999:
            print(f"\n🎉 PERFECT IoU! Stopping early.")
            break
    
    # Load best and evaluate
    model.load_checkpoint(checkpoint_path)
    
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}\n")
    
    _, train_iou = evaluate(model, train_loader, criterion, device)
    _, val_iou = evaluate(model, val_loader, criterion, device)
    _, test_iou = evaluate(model, test_loader, criterion, device)
    
    print(f"Train IoU: {train_iou:.4f}")
    print(f"Val IoU:   {val_iou:.4f}")
    print(f"Test IoU:  {test_iou:.4f}")
    print(f"\nGeneralization gap (Train-Test): {(train_iou - test_iou):.4f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Checkpoint: {checkpoint_path}")
    print(f"✓ Best epoch: {best_epoch}")
    print(f"✓ Best val IoU: {best_val_iou:.4f}")
    print(f"✓ Test IoU: {test_iou:.4f}")
    
    generalizes_well = test_iou > 0.7 and (train_iou - test_iou) < 0.2
    
    if generalizes_well:
        print("\n✅ MODEL GENERALIZES WELL!")
        print("   Ready for integration with the Brain!")
    else:
        print("\n⚠️  Model needs improvement")
        if test_iou < 0.7:
            print("   - Test IoU < 70%")
        if (train_iou - test_iou) > 0.2:
            print("   - Overfitting detected (gap > 20%)")
    
    return model, {
        'train_iou': train_iou,
        'val_iou': val_iou,
        'test_iou': test_iou,
        'best_epoch': best_epoch,
        'generalizes_well': generalizes_well
    }


if __name__ == "__main__":
    model, results = train_color_aware_object_cognition()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}\n")
    
    if results['generalizes_well']:
        print("🎉 COLOR-AWARE OBJECT COGNITION TRAINED SUCCESSFULLY!")
        print("\nSupported tasks:")
        for i in range(TaskType.NUM_TASK_TYPES):
            print(f"  {i:2d}. {TaskType.name(i)}")
    else:
        print("⚠️  Review generalization report above.")
