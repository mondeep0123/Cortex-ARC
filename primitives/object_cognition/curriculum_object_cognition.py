"""
Curriculum Task Generator for Object Cognition

Generates diverse tasks to train the object cognition primitive:
- Various object counts (0-15)
- Different object shapes (squares, lines, L-shapes, etc.)
- Different colors
- Different sizes
- Overlapping vs non-overlapping
- Different backgrounds

Goal: Model learns GENERAL object concept, not specific patterns
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import random


class ObjectCognitionCurriculumDataset(Dataset):
    """
    Dataset of synthetic object cognition tasks.
    
    Each task: Input grid →  Output count + masks
    """
    
    def __init__(self, num_tasks=1000, grid_size_range=(5, 15), seed=42):
        self.num_tasks = num_tasks
        self.grid_size_range = grid_size_range
        self.rng = np.random.RandomState(seed)
        
        # Pre-generate all tasks
        self.tasks = [self._generate_task() for _ in range(num_tasks)]
    
    def _generate_task(self) -> Tuple[np.ndarray, Dict]:
        """Generate one object cognition task (FIXED VERSION)."""
        
        # Random grid size
        h = self.rng.randint(*self.grid_size_range)
        w = self.rng.randint(*self.grid_size_range)
        
        # Background color (always 0)
        bg_color = 0
        grid = np.full((h, w), bg_color, dtype=np.int32)
        
        # Random number of objects to place (just for generation, not a target!)
        num_objects = self.rng.randint(0, 8)  # 0-7 objects
        
        # FIX 2: Place objects more carefully to avoid shortcuts
        objects_placed = 0
        max_attempts = 100  # More attempts for better placement
        
        # Track placed objects to avoid overlap
        placed_regions = []
        
        for _ in range(max_attempts):
            if objects_placed >= num_objects:
                break
            
            # Random object properties
            obj_color = self.rng.randint(1, 10)  # Non-background
            
            # FIX 3: More diverse object types
            obj_type = self.rng.choice([
                'single', 'single', 'single',  # More singles for diversity
                'square', 'line_h', 'line_v', 
                'L_shape', 'T_shape', 'plus'
            ])
            
            # Generate object
            if obj_type == 'single':
                obj_grid = np.array([[obj_color]])
            elif obj_type == 'square':
                size = self.rng.randint(2, 4)
                obj_grid = np.full((size, size), obj_color)
            elif obj_type == 'line_h':
                length = self.rng.randint(2, 5)
                obj_grid = np.full((1, length), obj_color)
            elif obj_type == 'line_v':
                length = self.rng.randint(2, 5)
                obj_grid = np.full((length, 1), obj_color)
            elif obj_type == 'L_shape':
                obj_grid = np.array([
                    [obj_color, 0],
                    [obj_color, obj_color]
                ])
            elif obj_type == 'T_shape':
                obj_grid = np.array([
                    [obj_color, obj_color, obj_color],
                    [0, obj_color, 0]
                ])
            elif obj_type == 'plus':
                obj_grid = np.array([
                    [0, obj_color, 0],
                    [obj_color, obj_color, obj_color],
                    [0, obj_color, 0]
                ])
            
            # Try to place
            oh, ow = obj_grid.shape
            if oh > h or ow > w:
                continue
            
            # Random position
            r = self.rng.randint(0, h - oh + 1)
            c = self.rng.randint(0, w - ow + 1)
            
            # Check overlap
            region = grid[r:r+oh, c:c+ow]
            if np.any(region != bg_color):
                continue  # Skip if overlaps
            
            # Place object (just color the grid, mask comes after!)
            for i in range(oh):
                for j in range(ow):
                    if obj_grid[i, j] != 0:
                        grid[r+i, c+j] = obj_color
            
            objects_placed += 1
        
        # NOW generate masks using SIMPLE RULE (like handcrafted test)
        # Rule: ANY non-zero cell = object
        segmentation_mask = (grid != 0).astype(np.float32)
        
        # Generate boundary mask (edges of objects)
        boundary_mask = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                if segmentation_mask[i, j] > 0:  # If this is object
                    # Check if adjacent to background
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if segmentation_mask[ni, nj] == 0:  # Adjacent to background
                                boundary_mask[i, j] = 1.0
                                break
        
        # Target: SEGMENTATION (not counting!)
        # Object Cognition = detect WHERE objects are
        target = {
            'segmentation': segmentation_mask,  # 1 for object, 0 for background
            'boundaries': boundary_mask   # 1 for edges, 0 elsewhere
        }
        
        return grid, target
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        grid, target = self.tasks[idx]
        
        # Convert to tensors
        grid_tensor = torch.from_numpy(grid).long()
        
        target_tensor = {
            'segmentation': torch.from_numpy(target['segmentation']),
            'boundaries': torch.from_numpy(target['boundaries'])
        }
        
        return grid_tensor, target_tensor




def collate_variable_grids(batch):
    """Custom collate to pad variable-sized grids."""
    grids, targets = zip(*batch)
    
    max_h = max(g.shape[0] for g in grids)
    max_w = max(g.shape[1] for g in grids)
    
    padded_grids = []
    padded_segs = []
    padded_bounds = []
    
    for grid, target in zip(grids, targets):
        h, w = grid.shape
        padded_grid = torch.zeros(max_h, max_w, dtype=grid.dtype)
        padded_grid[:h, :w] = grid
        padded_grids.append(padded_grid)
        
        seg = target['segmentation']
        padded_seg = torch.zeros(max_h, max_w, dtype=seg.dtype)
        padded_seg[:h, :w] = seg
        padded_segs.append(padded_seg)
        
        bound = target['boundaries']
        padded_bound = torch.zeros(max_h, max_w, dtype=bound.dtype)
        padded_bound[:h, :w] = bound
        padded_bounds.append(padded_bound)
    
    return torch.stack(padded_grids), {
        'segmentation': torch.stack(padded_segs),
        'boundaries': torch.stack(padded_bounds)
    }


def create_curriculum_loaders(
    train_size=7000,
    val_size=1500,
    test_size=1500,
    batch_size=32,
    grid_size_range=(5, 15)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test loaders for object cognition curriculum.
    
    Different random seeds ensure no overlap between sets!
    """
    
    print("Generating Object Cognition Curriculum...")
    print(f"  Train: {train_size} tasks")
    print(f"  Val:   {val_size} tasks")
    print(f"  Test:  {test_size} tasks")
    print(f"  Total: {train_size + val_size + test_size} tasks")
    
    # Create datasets with different seeds
    train_dataset = ObjectCognitionCurriculumDataset(
        num_tasks=train_size,
        grid_size_range=grid_size_range,
        seed=42
    )
    
    val_dataset = ObjectCognitionCurriculumDataset(
        num_tasks=val_size,
        grid_size_range=grid_size_range,
        seed=43  # Different seed!
    )
    
    test_dataset = ObjectCognitionCurriculumDataset(
        num_tasks=test_size,
        grid_size_range=grid_size_range,
        seed=44  # Different seed!
    )
    
    # Create loaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_grids
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable_grids
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable_grids
    )
    
    print("✓ Curriculum generated!")
    
    return train_loader, val_loader, test_loader


def visualize_sample_tasks(dataset, num_samples=5):
    """Visualize some curriculum tasks."""
    print("\n" + "="*60)
    print("SAMPLE CURRICULUM TASKS")
    print("="*60)
    
    for i in range(min(num_samples, len(dataset))):
        grid, target = dataset[i]
        grid_np = grid.numpy()
        
        print(f"\nTask {i+1}:")
        print(f"  Grid size: {grid_np.shape}")
        print(f"  Object cells: {target['segmentation'].sum().item():.0f}")
        print(f"  Boundary cells: {target['boundaries'].sum().item():.0f}")
        print(f"  Grid:")
        
        # Simple visualization
        for row in grid_np:
            print(f"    {' '.join(str(cell) for cell in row)}")


if __name__ == "__main__":
    # Test curriculum generation
    print("Testing Object Cognition Curriculum Generator")
    print("="*60)
    
    # Create small test dataset
    dataset = ObjectCognitionCurriculumDataset(num_tasks=10, grid_size_range=(5, 10))
    
    # Show samples
    visualize_sample_tasks(dataset, num_samples=5)
    
    # Create full loaders
    print("\n" + "="*60)
    train_loader, val_loader, test_loader = create_curriculum_loaders(
        train_size=100,  # Small for testing
        val_size=20,
        test_size=20,
        batch_size=8
    )
    
    # Test batch
    for batch in train_loader:
        grids, targets = batch
        print(f"\nBatch shape:")
        print(f"  Grids: {grids.shape}")
        print(f"  Segmentation: {targets['segmentation'].shape}")
        print(f"  Boundaries: {targets['boundaries'].shape}")
        break
    
    print("\n✓ Curriculum generator working!")
