"""
Color-Aware Object Cognition v3: Compositional Approach

PHILOSOPHY:
- Neural Net handles FUZZY tasks: Color selection, pattern extraction
- Algorithms handle RIGID tasks: Counting, sorting, topology

TASKS FOR NEURAL NET (this model):
1.  mask_all           - Mask all foreground
2.  mask_color_1       - Mask only color 1
3.  mask_color_2       - Mask only color 2
...
10. mask_color_9       - Mask only color 9

TASKS FOR COMPOSITION (use this + Numerosity):
- mask_dominant   = mask_color(argmax(count_each_color))
- mask_rare       = mask_color(argmin(count_each_color))
- mask_largest    = label_regions → count each → mask(argmax)

CHECKPOINT: checkpoints/color_object_cognition_v3.pt
(Separate from object_cognition_best.pt and color_aware_object_cognition_best.pt)

Author: Cortex-ARC Team
Date: December 26, 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
import numpy as np
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass  
class ColorCognitionConfig:
    """Configuration for Color Object Cognition v3."""
    name: str = "color_object_cognition_v3"
    base_channels: int = 64
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    device: str = 'cuda'


# =============================================================================
# TASK TYPES: ONLY COLOR SELECTION (10 tasks)
# =============================================================================

class TaskType:
    """
    Simplified task types: ONLY color selection.
    No counting, no statistics, no topology.
    Those are handled by Numerosity & scipy.
    """
    
    MASK_ALL = 0        # Mask all foreground (color > 0)
    MASK_COLOR_1 = 1    # Mask color 1
    MASK_COLOR_2 = 2    # Mask color 2
    MASK_COLOR_3 = 3    # Mask color 3
    MASK_COLOR_4 = 4    # Mask color 4
    MASK_COLOR_5 = 5    # Mask color 5
    MASK_COLOR_6 = 6    # Mask color 6
    MASK_COLOR_7 = 7    # Mask color 7
    MASK_COLOR_8 = 8    # Mask color 8
    MASK_COLOR_9 = 9    # Mask color 9
    
    NUM_TASKS = 10  # Only 10 tasks!
    
    @classmethod
    def name(cls, task_id: int) -> str:
        if task_id == 0:
            return "mask_all"
        return f"mask_color_{task_id}"


# =============================================================================
# MODEL: Simplified Color U-Net
# =============================================================================

class SpatialConvBlock(nn.Module):
    """Spatial-preserving convolutional block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ColorObjectCognition(nn.Module):
    """
    Color Object Cognition v3: U-Net with Task Conditioning.
    
    ONLY learns to mask by color. No statistics, no counting.
    """
    
    def __init__(self, config: ColorCognitionConfig):
        super().__init__()
        self.config = config
        base = config.base_channels
        
        # Color embedding (10 ARC colors → features)
        self.color_embedding = nn.Embedding(10, base)
        
        # Task embedding (10 masking tasks → conditioning)
        self.task_embedding = nn.Embedding(TaskType.NUM_TASKS, 32)
        
        # U-Net Encoder
        self.enc1 = SpatialConvBlock(base, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = SpatialConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck with FiLM task conditioning
        self.bottleneck = SpatialConvBlock(128, 256)
        self.task_proj = nn.Linear(32, 256 * 2) # Scale and Bias for FiLM
        
        # U-Net Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = SpatialConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = SpatialConvBlock(128, 64)
        
        # Output head (Logits)
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 1, 1)
        )
        
        self.optimizer = None
        self.scheduler = None
    
    def _pad_to_8(self, x):
        """Pad to multiple of 8 for pooling."""
        if x.dim() == 2:
            h, w = x.shape
            b = 1
        else:
            b, h, w = x.shape
        
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0)
        
        return x, (h, w)
    
    def forward(self, grid: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: [B, H, W] color indices (0-9)
            task: [B] task type (0-9)
        Returns:
            mask: [B, 1, H, W] binary mask probabilities
        """
        grid_pad, orig_size = self._pad_to_8(grid)
        h, w = orig_size
        
        # Embed colors
        x = self.color_embedding(grid_pad.long())
        x = x.permute(0, 3, 1, 2)
        
        # Encoder
        e1 = self.enc1(x)
        x = self.pool1(e1)
        
        e2 = self.enc2(x)
        x = self.pool2(e2)
        
        # Bottleneck + FiLM task conditioning
        x = self.bottleneck(x)
        task_emb = self.task_embedding(task)
        task_cond = self.task_proj(task_emb).unsqueeze(-1).unsqueeze(-1)
        
        # Split into scale and bias
        scale, bias = torch.chunk(task_cond, 2, dim=1)
        x = x * torch.sigmoid(scale) + bias
        
        # Decoder with skip connections
        x = self.up2(x)
        if x.shape[2:] != e2.shape[2:]:
            x = F.interpolate(x, size=e2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        if x.shape[2:] != e1.shape[2:]:
            x = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        
        # Output
        out = self.head(x)
        return out[:, :, :h, :w]
    
    # =========================================================================
    # CONVENIENCE METHODS FOR THE BRAIN
    # =========================================================================
    
    def mask_all(self, grid: torch.Tensor) -> torch.Tensor:
        """Mask all foreground pixels."""
        task = torch.zeros(grid.size(0), dtype=torch.long, device=grid.device)
        return self.forward(grid, task)
    
    def mask_color(self, grid: torch.Tensor, color: int) -> torch.Tensor:
        """Mask specific color (1-9)."""
        assert 1 <= color <= 9, f"Color must be 1-9, got {color}"
        task = torch.full((grid.size(0),), color, dtype=torch.long, device=grid.device)
        return self.forward(grid, task)
    
    # =========================================================================
    # COMPOSITIONAL METHODS (use Numerosity for counting)
    # =========================================================================
    
    def mask_dominant(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Mask the most frequent color.
        Uses counting to find dominant, then mask_color.
        """
        B = grid.size(0)
        results = []
        
        for b in range(B):
            g = grid[b]
            fg = g[g > 0]
            
            if len(fg) == 0:
                results.append(torch.zeros_like(g).unsqueeze(0).unsqueeze(0).float())
            else:
                colors, counts = torch.unique(fg, return_counts=True)
                dominant = colors[counts.argmax()].item()
                mask = self.mask_color(g.unsqueeze(0), int(dominant))
                results.append(mask)
        
        return torch.cat(results, dim=0)
    
    def mask_rare(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Mask the least frequent color.
        Uses counting to find rare, then mask_color.
        """
        B = grid.size(0)
        results = []
        
        for b in range(B):
            g = grid[b]
            fg = g[g > 0]
            
            if len(fg) == 0:
                results.append(torch.zeros_like(g).unsqueeze(0).unsqueeze(0).float())
            else:
                colors, counts = torch.unique(fg, return_counts=True)
                rare = colors[counts.argmin()].item()
                mask = self.mask_color(g.unsqueeze(0), int(rare))
                results.append(mask)
        
        return torch.cat(results, dim=0)
    
    # =========================================================================
    # TRAINING UTILITIES
    # =========================================================================
    
    def setup_training(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.learning_rate, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.max_epochs, eta_min=1e-6
        )
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': self.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.config.device, weights_only=False)
        self.load_state_dict(ckpt['model'])


# =============================================================================
# CURRICULUM DATASET
# =============================================================================

class ColorCurriculumDataset(Dataset):
    """
    STAGED Curriculum for Color Object Cognition.
    
    STAGES (like Numerosity training):
    - Stage 1 (Easy): Large grids 10-20, dense blocks, single target color
    - Stage 2 (Medium): Medium grids 8-15, lines/patterns, multi-color
    - Stage 3 (Hard): Small grids 3-10, frames/diagonals, sparse pixels
    """
    
    def __init__(self, num_samples: int, stage: int = 1, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.samples = []
        self.stage = stage
        
        # Stage-specific settings
        if stage == 1:
            self.grid_size_range = (5, 15) # Was 10-20
            self.complexity = 'easy'
        elif stage == 2:
            self.grid_size_range = (4, 12) # Was 8-15
            self.complexity = 'medium'
        else:  # stage 3
            self.grid_size_range = (3, 10) # Was 3-12
            self.complexity = 'hard'
        
        print(f"Generating {num_samples} Stage {stage} ({self.complexity}) samples...")
        for i in range(num_samples):
            self.samples.append(self._generate())
            if (i + 1) % 2500 == 0:
                print(f"  {i+1}/{num_samples}")
    
    def _generate(self):
        # Grid size based on stage
        H = self.rng.randint(self.grid_size_range[0], self.grid_size_range[1] + 1)
        W = self.rng.randint(self.grid_size_range[0], self.grid_size_range[1] + 1)
        
        # BALANCED TASK SELECTION: 10% for each task (0-9)
        task = self.rng.randint(0, 10)
        
        grid = np.zeros((H, W), dtype=np.int64)
        
        if task == 0:
            # mask_all: Generate grid with colors
            self._fill_grid_multi_color(grid, H, W)
            mask = (grid > 0).astype(np.float32)
        else:
            # mask_color_X: Task is the target color (1-9)
            target_color = task
            
            # Negative example rate based on stage (more negatives in later stages)
            neg_rate = 0.05 if self.stage == 1 else (0.10 if self.stage == 2 else 0.15)
            
            if self.rng.random() < neg_rate:
                # Fill with OTHER colors only
                other_colors = [c for c in range(1, 10) if c != target_color]
                self._fill_grid_with_colors(grid, H, W, other_colors)
                mask = np.zeros((H, W), dtype=np.float32)
            else:
                # POSITIVE: Target color GUARANTEED present
                # Fewer distracting colors in stage 1
                num_other = 1 if self.stage == 1 else self.rng.randint(1, 4)
                other_colors = [c for c in range(1, 10) if c != target_color]
                other_colors = self.rng.choice(other_colors, size=min(num_other, len(other_colors)), replace=False).tolist()
                all_colors = [target_color] + other_colors
                
                # DENSITY based on stage: Stage 1 more dense, Stage 3 more sparse
                if self.stage == 1:
                    density_type = self.rng.choice(['medium', 'dense', 'dense'])
                elif self.stage == 2:
                    density_type = self.rng.choice(['sparse', 'medium', 'dense'])
                else:
                    density_type = self.rng.choice(['sparse', 'sparse', 'medium'])
                
                # LOCATION based on stage
                if self.stage == 1:
                    location_type = self.rng.choice(['uniform', 'clustered', 'center', 'random_shape'])
                elif self.stage == 2:
                    location_type = self.rng.choice(['uniform', 'clustered', 'corners', 'center', 'line', 'random_shape'])
                else:
                    location_type = self.rng.choice(['uniform', 'clustered', 'corners', 'edges', 'center', 'line', 'diagonal', 'random_shape'])
                
                self._fill_grid_structured(grid, H, W, all_colors, target_color, density_type, location_type)
                mask = (grid == target_color).astype(np.float32)
        
        return grid, task, mask
    
    def _fill_grid_multi_color(self, grid, H, W):
        """Fill grid with multiple colors for mask_all task."""
        num_colors = self.rng.randint(1, 6)
        colors = self.rng.choice(range(1, 10), size=num_colors, replace=False).tolist()
        
        # Pattern type based on stage
        if self.stage == 1:
            pattern = self.rng.choice(['noise', 'blocks', 'blocks'])  # Simple
        elif self.stage == 2:
            pattern = self.rng.choice(['noise', 'blocks', 'lines', 'checkerboard'])  # Medium
        else:
            pattern = self.rng.choice(['noise', 'blocks', 'lines', 'checkerboard', 'frame', 'scattered'])  # All
        
        if pattern == 'noise':
            density = self.rng.uniform(0.2, 0.8)
            for r in range(H):
                for c in range(W):
                    if self.rng.random() < density:
                        grid[r, c] = self.rng.choice(colors)
        
        elif pattern == 'blocks':
            num_blocks = self.rng.randint(1, 4)
            for _ in range(num_blocks):
                color = self.rng.choice(colors)
                bh = self.rng.randint(1, max(2, H // 2))
                bw = self.rng.randint(1, max(2, W // 2))
                r0 = self.rng.randint(0, max(1, H - bh))
                c0 = self.rng.randint(0, max(1, W - bw))
                grid[r0:r0+bh, c0:c0+bw] = color
        
        elif pattern == 'lines':
            num_lines = self.rng.randint(1, 5)
            for _ in range(num_lines):
                color = self.rng.choice(colors)
                if self.rng.random() < 0.5:
                    row = self.rng.randint(0, H)
                    grid[row, :] = color
                else:
                    col = self.rng.randint(0, W)
                    grid[:, col] = color
        
        elif pattern == 'checkerboard':
            color = self.rng.choice(colors)
            offset = self.rng.randint(0, 2)
            for r in range(H):
                for c in range(W):
                    if (r + c + offset) % 2 == 0:
                        grid[r, c] = color
        
        elif pattern == 'frame':
            color = self.rng.choice(colors)
            grid[0, :] = color
            grid[-1, :] = color
            grid[:, 0] = color
            grid[:, -1] = color
        
        else:  # scattered
            num_pixels = self.rng.randint(1, H * W // 2)
            positions = self.rng.choice(H * W, size=num_pixels, replace=False)
            for pos in positions:
                r, c = pos // W, pos % W
                grid[r, c] = self.rng.choice(colors)
    
    def _fill_grid_with_colors(self, grid, H, W, colors):
        """Fill grid with specified colors only."""
        if len(colors) == 0:
            return
        density = self.rng.uniform(0.2, 0.6)
        for r in range(H):
            for c in range(W):
                if self.rng.random() < density:
                    grid[r, c] = self.rng.choice(colors)
    
    def _fill_grid_structured(self, grid, H, W, all_colors, target_color, density_type, location_type):
        """Fill grid with structured placement for target color."""
        
        # Determine target density
        if density_type == 'sparse':
            target_count = self.rng.randint(1, max(2, H * W // 10))
        elif density_type == 'medium':
            target_count = self.rng.randint(H * W // 10, H * W // 3)
        else:  # dense
            target_count = self.rng.randint(H * W // 3, H * W // 2)
        
        target_count = min(target_count, H * W - 1)
        
        # Place target color based on location type
        if location_type == 'uniform':
            positions = self.rng.choice(H * W, size=target_count, replace=False)
        elif location_type == 'clustered':
            # Cluster around a random center
            center_r = self.rng.randint(0, H)
            center_c = self.rng.randint(0, W)
            all_pos = [(r, c) for r in range(H) for c in range(W)]
            # Sort by distance to center
            all_pos.sort(key=lambda x: abs(x[0] - center_r) + abs(x[1] - center_c))
            positions = [p[0] * W + p[1] for p in all_pos[:target_count]]
        elif location_type == 'corners':
            # Prefer corners
            corner_positions = []
            for r in range(min(H//2+1, H)):
                for c in range(min(W//2+1, W)):
                    corner_positions.append(r * W + c)
            for r in range(max(0, H//2), H):
                for c in range(max(0, W//2), W):
                    if r * W + c not in corner_positions:
                        corner_positions.append(r * W + c)
            positions = self.rng.choice(corner_positions[:min(len(corner_positions), target_count*2)], 
                                        size=min(target_count, len(corner_positions)), replace=False)
        elif location_type == 'edges':
            # Only edges
            edge_positions = []
            for c in range(W):
                edge_positions.append(0 * W + c)
                edge_positions.append((H-1) * W + c)
            for r in range(1, H-1):
                edge_positions.append(r * W + 0)
                edge_positions.append(r * W + (W-1))
            edge_positions = list(set(edge_positions))
            positions = self.rng.choice(edge_positions, size=min(target_count, len(edge_positions)), replace=False)
        elif location_type == 'line':
            # Create a line
            if self.rng.random() < 0.5: # Horizontal
                r = self.rng.randint(0, H)
                positions = [r * W + c for c in range(W)]
            else: # Vertical
                c = self.rng.randint(0, W)
                positions = [r * W + c for r in range(H)]
        elif location_type == 'diagonal':
            # Create a diagonal
            if self.rng.random() < 0.5: # Main
                positions = [i * W + i for i in range(min(H, W))]
            else: # Anti
                positions = [i * W + (W - 1 - i) for i in range(min(H, W))]
        elif location_type == 'random_shape':
            # Create a random small shape or walk
            num_pixels = self.rng.randint(1, max(3, H*W//5))
            r, c = self.rng.randint(0, H), self.rng.randint(0, W)
            positions = [r * W + c]
            for _ in range(num_pixels - 1):
                directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]
                idx = self.rng.randint(0, len(directions))
                dr, dc = directions[idx]
                r, c = max(0, min(H-1, r+dr)), max(0, min(W-1, c+dc))
                positions.append(r * W + c)
            positions = list(set(positions))
        else:  # center
            # Prefer center
            center_r, center_c = H // 2, W // 2
            all_pos = [(r, c) for r in range(H) for c in range(W)]
            all_pos.sort(key=lambda x: abs(x[0] - center_r) + abs(x[1] - center_c))
            positions = [p[0] * W + p[1] for p in all_pos[:target_count]]
        
        # Place target color
        for pos in positions:
            r, c = pos // W, pos % W
            if 0 <= r < H and 0 <= c < W:
                grid[r, c] = target_color
        
        # Fill remaining with other colors
        other_colors = [c for c in all_colors if c != target_color]
        if other_colors:
            # BROAD DENSITY RANGE: from sparse to 100% full
            if self.stage == 1:
                other_density = self.rng.uniform(0.1, 0.4)
            elif self.stage == 2:
                other_density = self.rng.uniform(0.0, 0.7)
            else:
                other_density = self.rng.uniform(0.0, 1.0)
                
            for r in range(H):
                for c in range(W):
                    if grid[r, c] == 0 and self.rng.random() < other_density:
                        grid[r, c] = self.rng.choice(other_colors)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        grid, task, mask = self.samples[idx]
        return (
            torch.tensor(grid, dtype=torch.long),
            torch.tensor(task, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32)
        )


def collate_fn(batch):
    grids, tasks, masks = zip(*batch)
    
    max_h = max(g.shape[0] for g in grids)
    max_w = max(g.shape[1] for g in grids)
    
    # Round to 8
    max_h = ((max_h + 7) // 8) * 8
    max_w = ((max_w + 7) // 8) * 8
    
    padded_grids = []
    padded_masks = []
    
    for g, m in zip(grids, masks):
        h, w = g.shape
        padded_grids.append(F.pad(g, (0, max_w - w, 0, max_h - h)))
        padded_masks.append(F.pad(m, (0, max_w - w, 0, max_h - h)))
    
    return (
        torch.stack(padded_grids),
        torch.stack(list(tasks)),
        torch.stack(padded_masks).unsqueeze(1)
    )


# =============================================================================
# HANDCRAFTED BENCHMARK
# =============================================================================

def create_handcrafted_benchmark() -> List[Dict]:
    """
    16 handcrafted tests for Color Object Cognition.
    Tests color selection accuracy.
    """
    tests = []
    
    # Test 1: mask_all - simple
    tests.append({
        'name': 'mask_all_simple',
        'grid': np.array([
            [0, 1, 0],
            [2, 0, 3],
            [0, 1, 0]
        ]),
        'task': 0,
        'expected': np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
    })
    
    # Test 2: mask_all - dense
    tests.append({
        'name': 'mask_all_dense',
        'grid': np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]),
        'task': 0,
        'expected': np.ones((3, 3))
    })
    
    # Test 3: mask_all - empty
    tests.append({
        'name': 'mask_all_empty',
        'grid': np.zeros((4, 4), dtype=np.int64),
        'task': 0,
        'expected': np.zeros((4, 4))
    })
    
    # Test 4: mask_color_1
    tests.append({
        'name': 'mask_color_1',
        'grid': np.array([
            [1, 2, 1],
            [3, 1, 4],
            [1, 5, 1]
        ]),
        'task': 1,
        'expected': np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ])
    })
    
    # Test 5: mask_color_2
    tests.append({
        'name': 'mask_color_2',
        'grid': np.array([
            [2, 0, 2],
            [0, 2, 0],
            [2, 0, 2]
        ]),
        'task': 2,
        'expected': np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ])
    })
    
    # Test 6: mask_color_3 in multi-color grid
    tests.append({
        'name': 'mask_color_3_multi',
        'grid': np.array([
            [1, 2, 3, 4],
            [3, 3, 3, 5],
            [6, 7, 8, 3]
        ]),
        'task': 3,
        'expected': np.array([
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 1]
        ])
    })
    
    # Test 7: mask_color single pixel
    tests.append({
        'name': 'mask_color_5_single',
        'grid': np.array([
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 0]
        ]),
        'task': 5,
        'expected': np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
    })
    
    # Test 8: mask_color not present (should be empty)
    tests.append({
        'name': 'mask_color_9_absent',
        'grid': np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ]),
        'task': 9,
        'expected': np.zeros((3, 3))
    })
    
    # Test 9: mask_all checkerboard
    tests.append({
        'name': 'mask_all_checkerboard',
        'grid': np.array([
            [1, 0, 2, 0],
            [0, 3, 0, 4],
            [5, 0, 6, 0],
            [0, 7, 0, 8]
        ]),
        'task': 0,
        'expected': np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
    })
    
    # Test 10: mask_color horizontal line
    tests.append({
        'name': 'mask_color_4_line',
        'grid': np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [4, 4, 4, 4, 4],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]),
        'task': 4,
        'expected': np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    })
    
    # Test 11: mask_color vertical line
    tests.append({
        'name': 'mask_color_6_vertical',
        'grid': np.array([
            [0, 0, 6, 0, 0],
            [0, 0, 6, 0, 0],
            [0, 0, 6, 0, 0],
            [0, 0, 6, 0, 0]
        ]),
        'task': 6,
        'expected': np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ])
    })
    
    # Test 12: mask_color corner
    tests.append({
        'name': 'mask_color_7_corner',
        'grid': np.array([
            [7, 7, 0, 0],
            [7, 7, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]),
        'task': 7,
        'expected': np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
    })
    
    # Test 13: mask_all frame
    tests.append({
        'name': 'mask_all_frame',
        'grid': np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]),
        'task': 0,
        'expected': np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ])
    })
    
    # Test 14: mask_color_8 diagonal
    tests.append({
        'name': 'mask_color_8_diagonal',
        'grid': np.array([
            [8, 0, 0, 0],
            [0, 8, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 8]
        ]),
        'task': 8,
        'expected': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    })
    
    # Test 15: mask_color mixed sizes
    tests.append({
        'name': 'mask_color_2_mixed',
        'grid': np.array([
            [2, 2, 2, 0, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 3, 3],
            [0, 0, 0, 0, 0, 3, 3, 3]
        ]),
        'task': 2,
        'expected': np.array([
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
    })
    
    # Test 16: large grid mask_all
    tests.append({
        'name': 'mask_all_large',
        'grid': np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 6, 0, 0, 0, 0, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [9, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]),
        'task': 0,
        'expected': (np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 6, 0, 0, 0, 0, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [9, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]) > 0).astype(np.float32)
    })
    
    return tests


@torch.no_grad()
def run_benchmark(model: ColorObjectCognition, device: str) -> float:
    """Run handcrafted benchmark."""
    model.eval()
    tests = create_handcrafted_benchmark()
    
    print(f"\n{'='*60}")
    print("HANDCRAFTED BENCHMARK (16 tests)")
    print(f"{'='*60}\n")
    
    correct = 0
    
    for t in tests:
        grid = torch.tensor(t['grid'], dtype=torch.long).unsqueeze(0).to(device)
        task = torch.tensor([t['task']], dtype=torch.long).to(device)
        expected = torch.tensor(t['expected'], dtype=torch.float32)
        
        logits = model(grid, task).squeeze().cpu()
        pred = torch.sigmoid(logits)
        pred_binary = (pred > 0.5).float()
        
        # Crop to original size
        h, w = expected.shape
        pred_binary = pred_binary[:h, :w]
        
        match = (pred_binary == expected).all().item()
        
        if match:
            correct += 1
            print(f"  ✓ {t['name']}")
        else:
            print(f"  ✗ {t['name']}")
            # Show diff
            diff = (pred_binary != expected).sum().item()
            print(f"      {diff} pixels wrong")
    
    acc = correct / len(tests)
    print(f"\n{'='*60}")
    print(f"RESULT: {correct}/{len(tests)} = {acc:.1%}")
    print(f"{'='*60}\n")
    
    return acc


@torch.no_grad()
def run_benchmark_silent(model: ColorObjectCognition, device: str) -> float:
    """Run handcrafted benchmark silently (no print output)."""
    model.eval()
    tests = create_handcrafted_benchmark()
    
    correct = 0
    for t in tests:
        grid = torch.tensor(t['grid'], dtype=torch.long).unsqueeze(0).to(device)
        task = torch.tensor([t['task']], dtype=torch.long).to(device)
        expected = torch.tensor(t['expected'], dtype=torch.float32)
        
        logits = model(grid, task).squeeze().cpu()
        pred = torch.sigmoid(logits)
        pred_binary = (pred > 0.5).float()
        
        h, w = expected.shape
        pred_binary = pred_binary[:h, :w]
        
        if (pred_binary == expected).all().item():
            correct += 1
    
    return correct / len(tests)


# =============================================================================
# TRAINING
# =============================================================================

def compute_iou(preds, target, threshold=0.5):
    # Apply sigmoid since we now use logits
    probs = torch.sigmoid(preds)
    pred_bin = (probs > threshold).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return 1.0 if union == 0 else (inter / union).item()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_iou, n = 0.0, 0.0, 0
    
    for grids, tasks, masks in loader:
        grids, tasks, masks = grids.to(device), tasks.to(device), masks.to(device)
        
        optimizer.zero_grad()
        preds = model(grids, tasks)
        loss = criterion(preds, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_iou += compute_iou(preds, masks)
        n += 1
    
    return total_loss / n, total_iou / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0
    
    for grids, tasks, masks in loader:
        grids, tasks, masks = grids.to(device), tasks.to(device), masks.to(device)
        preds = model(grids, tasks)
        loss = criterion(preds, masks)
        total_loss += loss.item()
        total_iou += compute_iou(preds, masks)
        n += 1
    
    return total_loss / n, total_iou / n


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  COLOR OBJECT COGNITION v3  ".center(58) + "█")
    print("█" + "  (STAGED Training)  ".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60 + "\n")
    
    config = ColorCognitionConfig(device=device)
    
    print(f"Tasks: {TaskType.NUM_TASKS} (color selection only)")
    print(f"Device: {device}")
    
    # Model
    model = ColorObjectCognition(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    # Training setup
    model.setup_training()
    # Weighted loss to combat under-segmentation (Recall issues)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))
    checkpoint_path = "checkpoints/color_object_cognition_v3.pt"
    
    print(f"\nCheckpoint: {checkpoint_path}")
    
    best_handcrafted = 0.0
    best_val = 0.0
    best_epoch = 0
    global_epoch = 0
    
    # STAGED TRAINING (like Numerosity)
    for stage in [1, 2, 3]:
        print("\n" + "="*60)
        print(f"STAGE {stage}: {['EASY', 'MEDIUM', 'HARD'][stage-1]}")
        print("="*60 + "\n")
        
        # Generate stage-specific data
        train_ds = ColorCurriculumDataset(10000, stage=stage, seed=42 + stage*100)
        val_ds = ColorCurriculumDataset(2000, stage=stage, seed=43 + stage*100)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)
        
        print(f"\n{'Epoch':<6} {'Train IoU':<12} {'Val IoU':<12} {'HC':<8} {'Status'}")
        print("-" * 55)
        # Per-stage training
        stage_patience = 0
        stage_best_val = 0.0
        epochs_per_stage = 30  # Max epochs per stage
        
        for epoch in range(epochs_per_stage):
            global_epoch += 1
            _, train_iou = train_epoch(model, train_loader, model.optimizer, criterion, device)
            _, val_iou = evaluate(model, val_loader, criterion, device)
            model.scheduler.step()
            
            # Run handcrafted benchmark silently
            handcrafted_acc = run_benchmark_silent(model, device)
            
            status = ""
            
            # Priority 1: Best handcrafted (the REAL metric)
            if handcrafted_acc > best_handcrafted:
                best_handcrafted = handcrafted_acc
                best_val = val_iou
                best_epoch = global_epoch
                model.save(checkpoint_path)
                status = "✓ HC BEST"
                # Reset stage patience if HC improves
                stage_patience = 0
            
            # Priority 2: Best val within stage
            elif val_iou > stage_best_val:
                stage_best_val = val_iou
                stage_patience = 0
                if handcrafted_acc >= best_handcrafted * 0.95:  # Allow slight HC dip for val improve
                    model.save(checkpoint_path)
                status = "✓ VAL"
            else:
                stage_patience += 1
            
            # Print progress
            hc_str = f"{int(handcrafted_acc*16)}/16"
            if epoch % 2 == 0 or status or handcrafted_acc > 0.8: # More frequent printing
                print(f"{epoch:<6} {train_iou:<12.4f} {val_iou:<12.4f} {hc_str:<8} {status}")
            
            # Perfect on handcrafted - ADVANCE STAGE or STOP
            if handcrafted_acc >= 1.0:
                print(f"\n🏆 STAGE {stage} PERFECT on HC! Advancing...")
                break
            
            # Stage early stopping (validation/HC plateau)
            if stage_patience >= 8: # More aggressive plateau detection
                print(f"\n  Stage {stage} plateau reached. Advancing...")
                break
    
    # After all stages - final evaluation
    model.load(checkpoint_path)
    
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (after all stages)")
    print(f"{'='*60}\n")
    
    test_ds = ColorCurriculumDataset(2000, stage=3, seed=44)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)
    _, test_iou = evaluate(model, test_loader, criterion, device)
    
    print(f"Best Handcrafted: {int(best_handcrafted*16)}/16 = {best_handcrafted:.1%}")
    print(f"Test IoU: {test_iou:.4f}")
    
    # Handcrafted benchmark
    bench_acc = run_benchmark(model, device)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Checkpoint: {checkpoint_path}")
    print(f"✓ Best epoch: {best_epoch}")
    print(f"✓ Benchmark: {bench_acc:.1%}")
    
    if test_iou > 0.9 and bench_acc > 0.9:
        print("\n🎉 EXCELLENT! Ready for Brain integration!")
    elif test_iou > 0.8:
        print("\n✅ Good. May improve with more training.")
    else:
        print("\n⚠️  Needs improvement.")
    
    return model


if __name__ == "__main__":
    train()
