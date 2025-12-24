"""
Staged Curriculum for Numerosity - Following Cognitive Development

Stage 1: Subitizing Only (1-4 objects)
Stage 2: Small Compositional (5-8 objects)  
Stage 3: Medium Compositional (9-16 objects)
Stage 4: Large Compositional (17-30 objects)

This mimics how humans learn counting!
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List


class StagedNumerosityCurriculum(Dataset):
    """
    Curriculum that progresses through counting stages.
    
    Start with subitizing (1-4) → compositional (5+)
    """
    
    def __init__(self, num_tasks=10000, stage=1, grid_size_range=(3, 10), seed=42):
        """
        Args:
            num_tasks: Number of tasks to generate
            stage: 1=subitizing, 2=small_comp, 3=medium_comp, 4=large_comp
            grid_size_range: (min, max) grid dimensions
            seed: Random seed
        """
        self.num_tasks = num_tasks
        self.stage = stage
        self.grid_size_range = grid_size_range
        self.rng = np.random.RandomState(seed)
        
        # Define count ranges per stage
        self.stage_ranges = {
            1: (0, 4),    # Subitizing: 0-4 objects
            2: (5, 8),    # Small compositional: 5-8
            3: (9, 16),   # Medium compositional: 9-16
            4: (17, 30)   # Large compositional: 17-30
        }
        
        self.tasks = self._generate_all_tasks()
        
    def _generate_all_tasks(self):
        """Generate all tasks for this stage."""
        tasks = []
        for _ in range(self.num_tasks):
            task = self._generate_task()
            tasks.append(task)
        return tasks
    
    def _generate_task(self):
        """Generate a single counting task for current stage."""
        min_count, max_count = self.stage_ranges[self.stage]
        
        # Grid size (adapt to stage requirements)
        if self.stage == 1:
            # Subitizing: small grids (3x3 to 5x5)
            h = self.rng.randint(3, 6)
            w = self.rng.randint(3, 6)
        elif self.stage == 4:
            # Large compositional: need bigger grids for 17-30 objects
            h = self.rng.randint(6, self.grid_size_range[1] + 1)
            w = self.rng.randint(6, self.grid_size_range[1] + 1)
        else:
            # Stages 2-3: medium grids
            h = self.rng.randint(self.grid_size_range[0], self.grid_size_range[1] + 1)
            w = self.rng.randint(self.grid_size_range[0], self.grid_size_range[1] + 1)
        
        max_cells = h * w
        
        # Number of objects (ensure we can fit them!)
        min_count, max_count = self.stage_ranges[self.stage]
        if max_cells < min_count:
            # Grid too small for this stage, skip edge case
            min_count = max_cells // 2
        
        num_fill = self.rng.randint(min_count, min(max_count + 1, max_cells + 1))
        
        # Create grid
        grid = np.zeros((h, w), dtype=np.int64)
        
        if num_fill > 0:
            # Random positions
            all_positions = [(i, j) for i in range(h) for j in range(w)]
            self.rng.shuffle(all_positions)
            positions = all_positions[:num_fill]
            
            # Color distribution for this stage
            if self.stage == 1:
                # Subitizing: mostly single color (easier)
                color_prob = self.rng.random()
                if color_prob < 0.6:
                    # Single color
                    color = self.rng.randint(1, 10)
                    colors = [color] * num_fill
                else:
                    # 2-3 colors
                    num_colors = self.rng.randint(2, 4)
                    base_colors = self.rng.choice(range(1, 10), size=num_colors, replace=False)
                    colors = [base_colors[i % num_colors] for i in range(num_fill)]
            else:
                # Compositional: more color variety
                num_colors = self.rng.randint(1, min(6, num_fill + 1))
                base_colors = self.rng.choice(range(1, 10), size=num_colors, replace=False)
                colors = [base_colors[i % num_colors] for i in range(num_fill)]
            
            self.rng.shuffle(colors)
            
            for (i, j), color in zip(positions, colors):
                grid[i, j] = color
        
        # Compute targets
        total_count = num_fill
        color_counts = np.zeros(10, dtype=np.int64)
        for color in range(1, 10):
            color_counts[color] = (grid == color).sum()
        
        # Max color (most frequent non-background)
        if total_count > 0:
            max_color = color_counts[1:].argmax() + 1
        else:
            max_color = 0
        
        return {
            'grid': grid,
            'total_count': total_count,
            'color_counts': color_counts,
            'max_color': max_color,
            'stage': self.stage
        }
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        return {
            'grid': torch.from_numpy(task['grid']).long(),
            'total_count': torch.tensor(task['total_count']).long(),
            'color_counts': torch.from_numpy(task['color_counts']).long(),
            'max_color': torch.tensor(task['max_color']).long()
        }


def create_staged_dataloaders(stage, batch_size=64, train_size=15000, val_size=3000, test_size=3000):
    """Create dataloaders for a specific curriculum stage."""
    
    print(f"\n{'='*60}")
    print(f"CURRICULUM STAGE {stage}")
    print(f"{'='*60}")
    
    stage_names = {
        1: "Subitizing (0-4 objects)",
        2: "Small Compositional (5-8 objects)",
        3: "Medium Compositional (9-16 objects)",
        4: "Large Compositional (17-30 objects)"
    }
    
    print(f"Stage: {stage_names[stage]}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples: {test_size}\n")
    
    # Create datasets
    train_dataset = StagedNumerosityCurriculum(
        num_tasks=train_size,
        stage=stage,
        grid_size_range=(3, 15),
        seed=42 + stage * 1000  # Different seed per stage
    )
    
    val_dataset = StagedNumerosityCurriculum(
        num_tasks=val_size,
        stage=stage,
        grid_size_range=(3, 15),
        seed=43 + stage * 1000
    )
    
    test_dataset = StagedNumerosityCurriculum(
        num_tasks=test_size,
        stage=stage,
        grid_size_range=(3, 15),
        seed=44 + stage * 1000
    )
    
    # Custom collate function
    def collate_fn(batch):
        # Find max dimensions
        max_h = max(item['grid'].shape[0] for item in batch)
        max_w = max(item['grid'].shape[1] for item in batch)
        
        # Pad all grids
        padded_grids = []
        for item in batch:
            h, w = item['grid'].shape
            padded = torch.zeros(max_h, max_w, dtype=torch.long)
            padded[:h, :w] = item['grid']
            padded_grids.append(padded)
        
        return {
            'grid': torch.stack(padded_grids),
            'total_count': torch.stack([item['total_count'] for item in batch]),
            'color_counts': torch.stack([item['color_counts'] for item in batch]),
            'max_color': torch.stack([item['max_color'] for item in batch])
        }
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the staged curriculum
    print("Testing Staged Numerosity Curriculum\n")
    
    for stage in [1, 2, 3, 4]:
        train_loader, val_loader, test_loader = create_staged_dataloaders(
            stage=stage,
            batch_size=5,
            train_size=100,
            val_size=20,
            test_size=20
        )
        
        # Show sample
        batch = next(iter(train_loader))
        print(f"Sample batch:")
        print(f"  Grid shapes: {batch['grid'].shape}")
        print(f"  Total counts: {batch['total_count'][:5].tolist()}")
        print(f"  Min count: {batch['total_count'].min().item()}")
        print(f"  Max count: {batch['total_count'].max().item()}")
        print()
