"""
Color Training Tasks - ENHANCED VERSION

Generates diverse training tasks with many parameter variations.
This teaches the model the CONCEPT, not just specific rules.

Task Categories:
1. Identity - Copy exactly
2. Mask - Keep color by various criteria
3. Fill - Fill with color by various criteria  
4. Recolor - Transform colors by various rules

Each category has MULTIPLE variations to learn the general concept.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Generator
import random


class ColorTaskGenerator:
    """
    Enhanced task generator with diverse parameter variations.
    
    Instead of one rule per task, generates many variations
    so model learns the CONCEPT, not specific rules.
    """
    
    def __init__(self, num_colors: int = 10, max_size: int = 10, min_size: int = 3):
        self.num_colors = num_colors
        self.max_size = max_size
        self.min_size = min_size
        
        # All available task variations
        self.task_variations = {
            'identity': ['identity'],
            'mask_color': [
                'mask_smallest',      # Keep smallest color value
                'mask_largest',       # Keep largest color value
                'mask_most_frequent', # Keep most common color
                'mask_least_frequent',# Keep rarest color
                'mask_random',        # Keep random color (model must learn from examples)
            ],
            'find_dominant': [
                'fill_most_frequent', # Fill with most common
                'fill_least_frequent',# Fill with rarest
                'fill_smallest',      # Fill with smallest value
                'fill_largest',       # Fill with largest value
            ],
            'recolor': [
                'swap_smallest_two',  # Swap two smallest colors
                'swap_largest_two',   # Swap two largest colors
                'replace_smallest_with_largest',  # One-way replace
                'replace_largest_with_smallest',
                'increment_colors',   # Each color +1 (mod 10)
                'decrement_colors',   # Each color -1 (mod 10)
            ],
        }
    
    def generate_batch(
        self, 
        batch_size: int, 
        task_type: str = "mixed",
        variation: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of training examples.
        
        Args:
            batch_size: Number of examples
            task_type: Type of task or "mixed" for random
            variation: Specific variation, or None for random
        """
        inputs = []
        targets = []
        
        task_types = list(self.task_variations.keys())
        
        for _ in range(batch_size):
            if task_type == "mixed":
                t = random.choice(task_types)
            else:
                t = task_type
            
            # Get variation
            if variation:
                v = variation
            else:
                v = random.choice(self.task_variations[t])
            
            inp, tgt = self._generate_single(v)
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
            
            inp_padded = np.pad(inp, ((0, pad_h), (0, pad_w)), constant_values=0)
            tgt_padded = np.pad(tgt, ((0, pad_h), (0, pad_w)), constant_values=0)
            
            padded_inputs.append(inp_padded)
            padded_targets.append(tgt_padded)
        
        return (
            torch.tensor(np.array(padded_inputs), dtype=torch.long),
            torch.tensor(np.array(padded_targets), dtype=torch.long)
        )
    
    def _generate_single(self, variation: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single training example for a specific variation."""
        
        # Identity
        if variation == 'identity':
            return self._task_identity()
        
        # Mask variations
        elif variation == 'mask_smallest':
            return self._task_mask_by_criterion('smallest')
        elif variation == 'mask_largest':
            return self._task_mask_by_criterion('largest')
        elif variation == 'mask_most_frequent':
            return self._task_mask_by_criterion('most_frequent')
        elif variation == 'mask_least_frequent':
            return self._task_mask_by_criterion('least_frequent')
        elif variation == 'mask_random':
            return self._task_mask_by_criterion('random')
        
        # Fill variations
        elif variation == 'fill_most_frequent':
            return self._task_fill_by_criterion('most_frequent')
        elif variation == 'fill_least_frequent':
            return self._task_fill_by_criterion('least_frequent')
        elif variation == 'fill_smallest':
            return self._task_fill_by_criterion('smallest')
        elif variation == 'fill_largest':
            return self._task_fill_by_criterion('largest')
        
        # Recolor variations
        elif variation == 'swap_smallest_two':
            return self._task_swap_colors('smallest_two')
        elif variation == 'swap_largest_two':
            return self._task_swap_colors('largest_two')
        elif variation == 'replace_smallest_with_largest':
            return self._task_replace_color('smallest', 'largest')
        elif variation == 'replace_largest_with_smallest':
            return self._task_replace_color('largest', 'smallest')
        elif variation == 'increment_colors':
            return self._task_shift_colors(1)
        elif variation == 'decrement_colors':
            return self._task_shift_colors(-1)
        
        else:
            raise ValueError(f"Unknown variation: {variation}")
    
    def _random_grid(self, density: float = 0.5) -> np.ndarray:
        """Generate a random grid with colors."""
        h = random.randint(self.min_size, self.max_size)
        w = random.randint(self.min_size, self.max_size)
        
        grid = np.zeros((h, w), dtype=np.int64)
        
        num_pixels = int(h * w * density)
        for _ in range(num_pixels):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            color = random.randint(1, self.num_colors - 1)
            grid[r, c] = color
        
        return grid
    
    def _get_color_stats(self, grid: np.ndarray) -> Dict:
        """Compute color statistics for a grid."""
        colors = grid.flatten()
        colors = colors[colors != 0]  # Exclude background
        
        if len(colors) == 0:
            return {'colors': [], 'counts': {}, 'sorted': []}
        
        unique, counts = np.unique(colors, return_counts=True)
        count_dict = dict(zip(unique, counts))
        sorted_by_value = sorted(unique)
        sorted_by_freq = sorted(unique, key=lambda c: count_dict[c], reverse=True)
        
        return {
            'colors': list(unique),
            'counts': count_dict,
            'sorted_by_value': sorted_by_value,
            'sorted_by_freq': sorted_by_freq,
        }
    
    # ============= TASK IMPLEMENTATIONS =============
    
    def _task_identity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Output same as input."""
        grid = self._random_grid()
        return grid, grid.copy()
    
    def _task_mask_by_criterion(self, criterion: str) -> Tuple[np.ndarray, np.ndarray]:
        """Keep only one color based on criterion."""
        grid = self._random_grid()
        stats = self._get_color_stats(grid)
        
        if not stats['colors']:
            return grid, grid.copy()
        
        # Select which color to keep
        if criterion == 'smallest':
            keep = min(stats['colors'])
        elif criterion == 'largest':
            keep = max(stats['colors'])
        elif criterion == 'most_frequent':
            keep = stats['sorted_by_freq'][0]
        elif criterion == 'least_frequent':
            keep = stats['sorted_by_freq'][-1]
        elif criterion == 'random':
            keep = random.choice(stats['colors'])
        else:
            keep = min(stats['colors'])
        
        output = np.where(grid == keep, keep, 0)
        return grid, output
    
    def _task_fill_by_criterion(self, criterion: str) -> Tuple[np.ndarray, np.ndarray]:
        """Fill all non-zero positions with color based on criterion."""
        grid = self._random_grid(density=0.7)
        stats = self._get_color_stats(grid)
        
        if not stats['colors']:
            return grid, grid.copy()
        
        # Select fill color
        if criterion == 'most_frequent':
            fill = stats['sorted_by_freq'][0]
        elif criterion == 'least_frequent':
            fill = stats['sorted_by_freq'][-1]
        elif criterion == 'smallest':
            fill = min(stats['colors'])
        elif criterion == 'largest':
            fill = max(stats['colors'])
        else:
            fill = stats['sorted_by_freq'][0]
        
        output = np.where(grid != 0, fill, 0)
        return grid, output
    
    def _task_swap_colors(self, which: str) -> Tuple[np.ndarray, np.ndarray]:
        """Swap two colors."""
        grid = self._random_grid()
        stats = self._get_color_stats(grid)
        
        if len(stats['colors']) < 2:
            return grid, grid.copy()
        
        sorted_colors = stats['sorted_by_value']
        
        if which == 'smallest_two':
            c1, c2 = sorted_colors[0], sorted_colors[1]
        elif which == 'largest_two':
            c1, c2 = sorted_colors[-1], sorted_colors[-2]
        else:
            c1, c2 = sorted_colors[0], sorted_colors[1]
        
        output = grid.copy()
        mask1 = grid == c1
        mask2 = grid == c2
        output[mask1] = c2
        output[mask2] = c1
        
        return grid, output
    
    def _task_replace_color(self, source: str, target: str) -> Tuple[np.ndarray, np.ndarray]:
        """Replace one color with another (one-way, not swap)."""
        grid = self._random_grid()
        stats = self._get_color_stats(grid)
        
        if len(stats['colors']) < 2:
            return grid, grid.copy()
        
        sorted_colors = stats['sorted_by_value']
        
        if source == 'smallest':
            src = sorted_colors[0]
        else:
            src = sorted_colors[-1]
        
        if target == 'largest':
            tgt = sorted_colors[-1]
        else:
            tgt = sorted_colors[0]
        
        output = grid.copy()
        output[grid == src] = tgt
        
        return grid, output
    
    def _task_shift_colors(self, shift: int) -> Tuple[np.ndarray, np.ndarray]:
        """Shift all colors by amount (wrapping 1-9)."""
        grid = self._random_grid()
        
        output = grid.copy()
        for color in range(1, self.num_colors):
            new_color = ((color - 1 + shift) % 9) + 1  # Wrap 1-9
            output[grid == color] = new_color
        
        return grid, output


class ColorDataLoader:
    """DataLoader for color training tasks."""
    
    def __init__(
        self, 
        batch_size: int = 32,
        num_colors: int = 10,
        task_type: str = "mixed",
        variation: str = None,
    ):
        self.batch_size = batch_size
        self.task_type = task_type
        self.variation = variation
        self.generator = ColorTaskGenerator(num_colors=num_colors)
    
    def __iter__(self) -> Generator:
        """Infinite generator of batches."""
        while True:
            yield self.generator.generate_batch(
                self.batch_size, 
                self.task_type,
                self.variation,
            )
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single batch."""
        return self.generator.generate_batch(
            self.batch_size,
            self.task_type,
            self.variation,
        )


# Testing
if __name__ == "__main__":
    print("Testing Enhanced ColorTaskGenerator...")
    print("=" * 60)
    
    gen = ColorTaskGenerator()
    
    # Show all variations
    print("\nAvailable variations:")
    for task, variations in gen.task_variations.items():
        print(f"  {task}:")
        for v in variations:
            print(f"    - {v}")
    
    # Test each variation
    print("\nTesting each variation...")
    for task, variations in gen.task_variations.items():
        print(f"\n{task}:")
        for v in variations:
            inp, tgt = gen._generate_single(v)
            match = np.array_equal(inp, tgt)
            print(f"  {v}: in={inp.shape}, out={tgt.shape}, same={match}")
    
    print("\n✓ Enhanced color tasks working!")
    print(f"Total variations: {sum(len(v) for v in gen.task_variations.values())}")
