"""
Color Training Tasks

Generates synthetic tasks to teach the model color understanding:
1. Color Identity: Given input, output same color pattern
2. Color Counting: Identify distinct color regions
3. Color Grouping: Group same-colored pixels
4. Dominant Color: Find most common color
5. Color Masking: Isolate specific colors

These tasks are simple but fundamental - the model must first
understand what "color" means before it can reason about
color-based patterns in ARC puzzles.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Generator
import random


class ColorTaskGenerator:
    """
    Generates training tasks for color understanding.
    
    Each task type teaches a specific aspect of color perception.
    All tasks use the same model - abilities accumulate.
    """
    
    def __init__(self, num_colors: int = 10, max_size: int = 10, min_size: int = 3):
        self.num_colors = num_colors
        self.max_size = max_size
        self.min_size = min_size
    
    def generate_batch(
        self, 
        batch_size: int, 
        task_type: str = "mixed"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of training examples.
        
        Args:
            batch_size: Number of examples
            task_type: Type of task or "mixed" for random
            
        Returns:
            (input_grids, target_grids) as tensors
        """
        inputs = []
        targets = []
        
        task_types = ["identity", "mask_color", "find_dominant", "recolor"]
        
        for _ in range(batch_size):
            if task_type == "mixed":
                t = random.choice(task_types)
            else:
                t = task_type
            
            inp, tgt = self._generate_single(t)
            inputs.append(inp)
            targets.append(tgt)
        
        # Pad to same size
        max_h = max(i.shape[0] for i in inputs)
        max_w = max(i.shape[1] for i in inputs)
        
        padded_inputs = []
        padded_targets = []
        
        for inp, tgt in zip(inputs, targets):
            h, w = inp.shape
            pad_h = max_h - h
            pad_w = max_w - w
            
            # Pad with 0 (background)
            inp_padded = np.pad(inp, ((0, pad_h), (0, pad_w)), constant_values=0)
            tgt_padded = np.pad(tgt, ((0, pad_h), (0, pad_w)), constant_values=0)
            
            padded_inputs.append(inp_padded)
            padded_targets.append(tgt_padded)
        
        return (
            torch.tensor(np.array(padded_inputs), dtype=torch.long),
            torch.tensor(np.array(padded_targets), dtype=torch.long)
        )
    
    def _generate_single(self, task_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single training example."""
        
        if task_type == "identity":
            return self._task_identity()
        elif task_type == "mask_color":
            return self._task_mask_color()
        elif task_type == "find_dominant":
            return self._task_find_dominant()
        elif task_type == "recolor":
            return self._task_recolor()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _random_grid(self, density: float = 0.5) -> np.ndarray:
        """Generate a random grid with colors."""
        h = random.randint(self.min_size, self.max_size)
        w = random.randint(self.min_size, self.max_size)
        
        grid = np.zeros((h, w), dtype=np.int64)
        
        # Randomly place colored pixels
        num_pixels = int(h * w * density)
        for _ in range(num_pixels):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            color = random.randint(1, self.num_colors - 1)  # 1-9, not 0
            grid[r, c] = color
        
        return grid
    
    def _task_identity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Task: Output the same grid as input.
        
        Purpose: Learn to encode and decode colors correctly.
        This is the most basic task - if the model can't do this,
        nothing else will work.
        """
        grid = self._random_grid()
        return grid, grid.copy()
    
    def _task_mask_color(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Task: Keep only one specific color, set others to 0.
        
        Purpose: Learn to identify and isolate colors.
        The model must understand "this pixel is color X" vs "not color X".
        """
        grid = self._random_grid()
        
        # Find colors present (excluding 0)
        colors_present = list(set(grid.flatten()) - {0})
        if len(colors_present) == 0:
            # Empty grid - just return identity
            return grid, grid.copy()
        
        # Always keep the SMALLEST color (deterministic - model can learn this)
        keep_color = min(colors_present)
        
        # Output: only that color, rest is 0
        output = np.where(grid == keep_color, keep_color, 0)
        
        return grid, output
    
    def _task_find_dominant(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Task: Fill entire grid with the most common color.
        
        Purpose: Learn to count colors and identify the dominant one.
        Requires understanding color frequency.
        """
        grid = self._random_grid(density=0.7)
        
        # Find most common non-zero color
        colors = grid.flatten()
        colors = colors[colors != 0]
        
        if len(colors) == 0:
            return grid, grid.copy()
        
        # Count occurrences
        unique, counts = np.unique(colors, return_counts=True)
        dominant = unique[np.argmax(counts)]
        
        # Output: fill with dominant color (only non-zero positions)
        output = np.where(grid != 0, dominant, 0)
        
        return grid, output
    
    def _task_recolor(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Task: Swap two colors.
        
        Purpose: Learn color relationships and transformations.
        If color A becomes B, every A must become B.
        """
        grid = self._random_grid()
        
        # Find colors present
        colors_present = list(set(grid.flatten()) - {0})
        if len(colors_present) < 2:
            return grid, grid.copy()
        
        # Always swap the two SMALLEST colors (deterministic)
        sorted_colors = sorted(colors_present)
        c1, c2 = sorted_colors[0], sorted_colors[1]
        
        # Swap
        output = grid.copy()
        mask1 = grid == c1
        mask2 = grid == c2
        output[mask1] = c2
        output[mask2] = c1
        
        return grid, output


class ColorDataLoader:
    """
    DataLoader for color training tasks.
    
    Generates batches on-the-fly (infinite data!).
    """
    
    def __init__(
        self, 
        batch_size: int = 32,
        num_colors: int = 10,
        task_type: str = "mixed"
    ):
        self.batch_size = batch_size
        self.task_type = task_type
        self.generator = ColorTaskGenerator(num_colors=num_colors)
    
    def __iter__(self) -> Generator:
        """Infinite generator of batches."""
        while True:
            yield self.generator.generate_batch(
                self.batch_size, 
                self.task_type
            )
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single batch."""
        return self.generator.generate_batch(
            self.batch_size,
            self.task_type
        )


# Testing
if __name__ == "__main__":
    print("Testing ColorTaskGenerator...")
    
    gen = ColorTaskGenerator()
    
    # Test each task type
    for task in ["identity", "mask_color", "find_dominant", "recolor"]:
        inp, tgt = gen.generate_batch(4, task)
        print(f"\n{task}:")
        print(f"  Input shape: {inp.shape}")
        print(f"  Target shape: {tgt.shape}")
        print(f"  Sample input:\n{inp[0].numpy()}")
        print(f"  Sample target:\n{tgt[0].numpy()}")
    
    # Test dataloader
    print("\n\nTesting ColorDataLoader...")
    loader = ColorDataLoader(batch_size=8)
    inp, tgt = loader.get_batch()
    print(f"Batch shapes: {inp.shape}, {tgt.shape}")
    
    print("\n✓ Color tasks working!")
