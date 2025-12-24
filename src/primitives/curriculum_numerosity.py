"""
Numerosity Curriculum Generator

Generates counting and comparison tasks.

Task types:
1. Total count: How many non-background cells?
2. Color counts: How many of each color?
3. Max color: Which color appears most?

Goal: Learn to COUNT, not just segment!
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List


class NumerosityCurriculumDataset(Dataset):
    """Generate numerosity (counting) tasks."""
    
    def __init__(
        self,
        num_tasks=10000,
        grid_size_range=(3, 15),
        max_count=30,
        seed=42
    ):
        self.num_tasks = num_tasks
        self.grid_size_range = grid_size_range
        self.max_count = max_count
        self.rng = np.random.RandomState(seed)
        
        print(f"Generating Numerosity Curriculum...")
        print(f"  Tasks: {num_tasks}")
        print(f"  Grid size: {grid_size_range}")
        print(f"  Max count: {max_count}")
        
        # Generate all tasks
        self.tasks = [self._generate_task() for _ in range(num_tasks)]
        print(f"✓ Curriculum generated!")
    
    def _generate_task(self) -> Tuple[np.ndarray, Dict]:
        """
        Generate one counting task.
        
        IMPROVED: Better distribution to match handcrafted benchmark!
        - Emphasize small grids (3x3, 5x5)
        - More low counts (1-10) 
        - Edge cases (empty, single, full)
        
        Returns grid and targets for counting.
        """
        # Improved grid size distribution
        # 40% small (3-6), 40% medium (7-10), 20% large (11-15)
        size_category = self.rng.choice(['small', 'medium', 'large'], p=[0.4, 0.4, 0.2])
        
        if size_category == 'small':
            h = self.rng.randint(3, 7)  # 3-6
            w = self.rng.randint(3, 7)
        elif size_category == 'medium':
            h = self.rng.randint(7, 11)  # 7-10
            w = self.rng.randint(7, 11)
        else:
            h = self.rng.randint(11, 16)  # 11-15
            w = self.rng.randint(11, 16)
        
        # Start with background
        grid = np.zeros((h, w), dtype=np.int32)
        
        # Improved count distribution
        # 20% edge cases, 50% low counts (1-10), 30% medium-high (11-30)
        case_type = self.rng.choice(['edge', 'low', 'high'], p=[0.2, 0.5, 0.3])
        
        max_cells = min(h * w, self.max_count)
        
        if case_type == 'edge':
            # Edge cases: empty, single cell, or full grid
            edge_case = self.rng.choice(['empty', 'single', 'full'])
            if edge_case == 'empty':
                num_fill = 0
            elif edge_case == 'single':
                num_fill = 1
            else:  # full
                num_fill = min(h * w, self.max_count)
        elif case_type == 'low':
            # Low counts: 1-10 (or up to max_cells if smaller)
            num_fill = self.rng.randint(1, min(11, max_cells + 1))
        else:
            # Medium-high counts: 11-30 (or max_cells if smaller)
            if max_cells >= 11:
                num_fill = self.rng.randint(11, max_cells + 1)
            else:
                # Grid too small for high counts, use low instead
                num_fill = self.rng.randint(1, max_cells + 1)
        
        if num_fill > 0:
            # Random positions
            positions = self.rng.choice(h * w, size=num_fill, replace=False)
            rows = positions // w
            cols = positions % w
            
            # Improved color distribution
            # 30% single color, 40% 2-3 colors, 30% many colors
            color_diversity = self.rng.choice(['single', 'few', 'many'], p=[0.3, 0.4, 0.3])
            
            if color_diversity == 'single':
                # All same color
                color = self.rng.randint(1, 10)
                colors = np.full(num_fill, color, dtype=np.int32)
            elif color_diversity == 'few':
                # 2-3 colors
                num_colors = self.rng.randint(2, 4)
                available_colors = self.rng.choice(range(1, 10), size=num_colors, replace=False)
                colors = self.rng.choice(available_colors, size=num_fill)
            else:
                # Many colors (random)
                colors = self.rng.randint(1, 10, size=num_fill)
            
            grid[rows, cols] = colors
        
        # Compute targets
        total_count = num_fill
        
        # Count per color
        color_counts = np.zeros(10, dtype=np.int64)
        for color in range(10):
            color_counts[color] = (grid == color).sum()
        
        # Most common non-background color
        if total_count > 0:
            non_bg_counts = color_counts[1:]  # Exclude background
            max_color = np.argmax(non_bg_counts) + 1  # +1 since we excluded 0
        else:
            max_color = 0  # Background if empty
        
        target = {
            'total_count': np.array(total_count, dtype=np.int64),
            'color_counts': color_counts,
            'max_color': np.array(max_color, dtype=np.int64)
        }
        
        return grid, target
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        grid, target = self.tasks[idx]
        
        # Convert to tensors
        grid_tensor = torch.from_numpy(grid).long()
        
        target_tensor = {
            'total_count': torch.from_numpy(target['total_count']),
            'color_counts': torch.from_numpy(target['color_counts']),
            'max_color': torch.from_numpy(target['max_color'])
        }
        
        return grid_tensor, target_tensor


def collate_numerosity(batch):
    """Custom collate to pad variable-sized grids."""
    grids, targets = zip(*batch)
    
    max_h = max(g.shape[0] for g in grids)
    max_w = max(g.shape[1] for g in grids)
    
    padded_grids = []
    
    for grid in grids:
        h, w = grid.shape
        padded_grid = torch.zeros(max_h, max_w, dtype=grid.dtype)
        padded_grid[:h, :w] = grid
        padded_grids.append(padded_grid)
    
    return torch.stack(padded_grids), {
        'total_count': torch.stack([t['total_count'] for t in targets]),
        'color_counts': torch.stack([t['color_counts'] for t in targets]),
        'max_color': torch.stack([t['max_color'] for t in targets])
    }


def create_numerosity_loaders(
    train_size=15000,
    val_size=3000,
    test_size=3000,
    batch_size=64,
    grid_size_range=(3, 15)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test loaders for numerosity.
    """
    print("Creating Numerosity Curriculum...")
    
    # Different seeds for train/val/test
    train_dataset = NumerosityCurriculumDataset(
        num_tasks=train_size,
        grid_size_range=grid_size_range,
        seed=100
    )
    
    val_dataset = NumerosityCurriculumDataset(
        num_tasks=val_size,
        grid_size_range=grid_size_range,
        seed=101
    )
    
    test_dataset = NumerosityCurriculumDataset(
        num_tasks=test_size,
        grid_size_range=grid_size_range,
        seed=102
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_numerosity
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_numerosity
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_numerosity
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test curriculum
    print("Testing Numerosity Curriculum")
    print("="*60)
    
    dataset = NumerosityCurriculumDataset(num_tasks=5, grid_size_range=(4, 8))
    
    print("\nSample tasks:")
    for i in range(5):
        grid, target = dataset[i]
        print(f"\nTask {i+1}:")
        print(f"  Grid shape: {grid.shape}")
        print(f"  Total count: {target['total_count'].item()}")
        print(f"  Max color: {target['max_color'].item()}")
        print(f"  Grid:")
        for row in grid.numpy():
            print(f"    {' '.join(str(x) for x in row)}")
    
    print("\n✓ Curriculum working!")
