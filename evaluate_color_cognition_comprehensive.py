"""
Comprehensive Evaluation of Color Object Cognition v3

Tests THREE categories:
1. LEARNED SKILLS (Neural Net alone): mask_all, mask_color_1-9
2. COMPOSITIONAL (Neural Net + Algorithms): mask_dominant, mask_rare
3. ALGORITHMIC (scipy.ndimage): mask_largest, mask_smallest, mask_isolated, mask_boundary

All tests use 8x8 or larger grids to avoid U-Net edge effects.

Author: Cortex-ARC Team
Date: December 26, 2025
"""

import torch
import numpy as np
from pathlib import Path
import sys
from scipy import ndimage
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from train_color_object_cognition_v3 import ColorObjectCognition, ColorCognitionConfig


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model(device='cuda'):
    config = ColorCognitionConfig(device=device)
    model = ColorObjectCognition(config).to(device)
    model.load("checkpoints/color_object_cognition_v3.pt")
    model.eval()
    return model


# =============================================================================
# ALGORITHMIC PRIMITIVES (100% accurate, no ML)
# =============================================================================

def algo_mask_dominant(grid: np.ndarray) -> np.ndarray:
    """Mask the most frequent color. Algorithmic, 100% accurate."""
    fg = grid[grid > 0]
    if len(fg) == 0:
        return np.zeros_like(grid, dtype=np.float32)
    colors, counts = np.unique(fg, return_counts=True)
    dominant = colors[np.argmax(counts)]
    return (grid == dominant).astype(np.float32)


def algo_mask_rare(grid: np.ndarray) -> np.ndarray:
    """Mask the least frequent color. Algorithmic, 100% accurate."""
    fg = grid[grid > 0]
    if len(fg) == 0:
        return np.zeros_like(grid, dtype=np.float32)
    colors, counts = np.unique(fg, return_counts=True)
    rare = colors[np.argmin(counts)]
    return (grid == rare).astype(np.float32)


def algo_mask_largest_region(grid: np.ndarray) -> np.ndarray:
    """Mask the largest connected component. Uses scipy.ndimage."""
    fg = (grid > 0).astype(np.int32)
    labeled, num_features = ndimage.label(fg)
    if num_features == 0:
        return np.zeros_like(grid, dtype=np.float32)
    sizes = ndimage.sum(fg, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label).astype(np.float32)


def algo_mask_smallest_region(grid: np.ndarray) -> np.ndarray:
    """Mask the smallest connected component. Uses scipy.ndimage."""
    fg = (grid > 0).astype(np.int32)
    labeled, num_features = ndimage.label(fg)
    if num_features == 0:
        return np.zeros_like(grid, dtype=np.float32)
    sizes = ndimage.sum(fg, labeled, range(1, num_features + 1))
    smallest_label = np.argmin(sizes) + 1
    return (labeled == smallest_label).astype(np.float32)


def algo_mask_isolated(grid: np.ndarray) -> np.ndarray:
    """Mask single pixels with no 4-connected neighbors."""
    fg = (grid > 0).astype(np.int32)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    neighbor_count = ndimage.convolve(fg, kernel, mode='constant', cval=0)
    isolated = fg * (neighbor_count == 0)
    return isolated.astype(np.float32)


def algo_mask_boundary(grid: np.ndarray) -> np.ndarray:
    """Mask foreground pixels on grid boundary."""
    H, W = grid.shape
    mask = np.zeros((H, W), dtype=np.float32)
    mask[0, :] = (grid[0, :] > 0)
    mask[-1, :] = (grid[-1, :] > 0)
    mask[:, 0] = (grid[:, 0] > 0)
    mask[:, -1] = (grid[:, -1] > 0)
    return mask


# =============================================================================
# COMPOSITIONAL: Use Neural Net + Algorithm
# =============================================================================

def compositional_mask_dominant(model, grid: np.ndarray, device: str) -> np.ndarray:
    """
    Mask dominant color using composition:
    1. Algorithm finds which color is dominant
    2. Neural net masks that color
    """
    # Step 1: Find dominant color (algorithm)
    fg = grid[grid > 0]
    if len(fg) == 0:
        return np.zeros_like(grid, dtype=np.float32)
    colors, counts = np.unique(fg, return_counts=True)
    dominant = int(colors[np.argmax(counts)])
    
    # Step 2: Use neural net to mask that color
    with torch.no_grad():
        g = torch.tensor(grid, dtype=torch.long).unsqueeze(0).to(device)
        t = torch.tensor([dominant], dtype=torch.long).to(device)
        pred = model(g, t).squeeze().cpu().numpy()
    
    h, w = grid.shape
    return (pred[:h, :w] > 0.5).astype(np.float32)


def compositional_mask_rare(model, grid: np.ndarray, device: str) -> np.ndarray:
    """
    Mask rare color using composition:
    1. Algorithm finds which color is rarest
    2. Neural net masks that color
    """
    fg = grid[grid > 0]
    if len(fg) == 0:
        return np.zeros_like(grid, dtype=np.float32)
    colors, counts = np.unique(fg, return_counts=True)
    rare = int(colors[np.argmin(counts)])
    
    with torch.no_grad():
        g = torch.tensor(grid, dtype=torch.long).unsqueeze(0).to(device)
        t = torch.tensor([rare], dtype=torch.long).to(device)
        pred = model(g, t).squeeze().cpu().numpy()
    
    h, w = grid.shape
    return (pred[:h, :w] > 0.5).astype(np.float32)


# =============================================================================
# HANDCRAFTED TEST GRIDS (8x8 minimum for U-Net compatibility)
# =============================================================================

def create_test_grids() -> List[Dict]:
    """
    Create test grids that work with U-Net (8x8 minimum).
    Each grid tests multiple skills.
    """
    tests = []
    
    # Test Grid 1: Multi-color with clear dominant/rare
    tests.append({
        'name': 'multi_color_stats',
        'grid': np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 2, 2, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 4, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]),
        'expected': {
            'mask_all': np.array([
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 1, 1, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'mask_color_1': np.array([
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'mask_color_2': np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'mask_dominant': np.array([  # Color 1 is dominant (16 pixels)
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'mask_rare': np.array([  # Colors 3 and 4 are tied (1 pixel each), pick first
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'mask_largest_region': np.array([  # The 4x4 block of 1s is largest
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'mask_smallest_region': np.array([  # Single pixels are smallest
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],  # This one picked first
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'mask_isolated': np.array([  # Single pixels with no neighbors
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ])
        }
    })
    
    # Test Grid 2: Checkerboard pattern
    tests.append({
        'name': 'checkerboard',
        'grid': np.array([
            [5, 0, 5, 0, 5, 0, 5, 0],
            [0, 5, 0, 5, 0, 5, 0, 5],
            [5, 0, 5, 0, 5, 0, 5, 0],
            [0, 5, 0, 5, 0, 5, 0, 5],
            [5, 0, 5, 0, 5, 0, 5, 0],
            [0, 5, 0, 5, 0, 5, 0, 5],
            [5, 0, 5, 0, 5, 0, 5, 0],
            [0, 5, 0, 5, 0, 5, 0, 5]
        ]),
        'expected': {
            'mask_all': (np.array([
                [5, 0, 5, 0, 5, 0, 5, 0],
                [0, 5, 0, 5, 0, 5, 0, 5],
                [5, 0, 5, 0, 5, 0, 5, 0],
                [0, 5, 0, 5, 0, 5, 0, 5],
                [5, 0, 5, 0, 5, 0, 5, 0],
                [0, 5, 0, 5, 0, 5, 0, 5],
                [5, 0, 5, 0, 5, 0, 5, 0],
                [0, 5, 0, 5, 0, 5, 0, 5]
            ]) > 0).astype(float),
            'mask_color_5': (np.array([
                [5, 0, 5, 0, 5, 0, 5, 0],
                [0, 5, 0, 5, 0, 5, 0, 5],
                [5, 0, 5, 0, 5, 0, 5, 0],
                [0, 5, 0, 5, 0, 5, 0, 5],
                [5, 0, 5, 0, 5, 0, 5, 0],
                [0, 5, 0, 5, 0, 5, 0, 5],
                [5, 0, 5, 0, 5, 0, 5, 0],
                [0, 5, 0, 5, 0, 5, 0, 5]
            ]) == 5).astype(float),
            'mask_isolated': (np.ones((8, 8)) * np.array([
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1]
            ])).astype(float)  # All are isolated since no 4-connected neighbors
        }
    })
    
    # Test Grid 3: Connected regions of different sizes
    tests.append({
        'name': 'regions',
        'grid': np.array([
            [2, 2, 2, 0, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0, 0],
            [2, 2, 2, 0, 3, 3, 0, 0],
            [0, 0, 0, 0, 3, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]),
        'expected': {
            'mask_largest_region': np.array([  # 3x3 = 9 pixels
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'mask_smallest_region': np.array([  # 1 pixel
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ])
        }
    })
    
    # Test Grid 4: Horizontal and vertical lines (fixed 8x8)
    tests.append({
        'name': 'lines',
        'grid': np.array([
            [0, 0, 0, 6, 0, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0, 0],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [0, 0, 0, 6, 0, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0, 0]
        ]),
        'expected': {
            'mask_color_6': np.array([
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],  # intersection is 7, not 6
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0]
            ]),
            'mask_color_7': np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'mask_boundary': np.array([  # Foreground pixels on edge
                [0, 0, 0, 1, 0, 0, 0, 0],  # top row: 6 at col 3
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1],  # left and right edge of row 3
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0]   # bottom row: 6 at col 3
            ])
        }
    })
    
    # Test Grid 5: Empty grid
    tests.append({
        'name': 'empty',
        'grid': np.zeros((8, 8), dtype=np.int64),
        'expected': {
            'mask_all': np.zeros((8, 8)),
            'mask_color_5': np.zeros((8, 8)),
            'mask_dominant': np.zeros((8, 8)),
            'mask_largest_region': np.zeros((8, 8))
        }
    })
    
    return tests


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_comprehensive(model, device):
    """Run comprehensive evaluation on all skills."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE COLOR OBJECT COGNITION EVALUATION")
    print("="*70)
    
    tests = create_test_grids()
    
    results = {
        'learned': {'pass': 0, 'fail': 0, 'details': []},
        'compositional': {'pass': 0, 'fail': 0, 'details': []},
        'algorithmic': {'pass': 0, 'fail': 0, 'details': []}
    }
    
    for test in tests:
        grid = test['grid']
        name = test['name']
        
        print(f"\n--- Grid: {name} ({grid.shape[0]}x{grid.shape[1]}) ---")
        
        for skill, expected in test['expected'].items():
            # Determine skill category
            if skill in ['mask_all', 'mask_color_1', 'mask_color_2', 'mask_color_3',
                        'mask_color_4', 'mask_color_5', 'mask_color_6', 'mask_color_7',
                        'mask_color_8', 'mask_color_9']:
                category = 'learned'
                # Use neural net
                task_id = 0 if skill == 'mask_all' else int(skill.split('_')[-1])
                with torch.no_grad():
                    g = torch.tensor(grid, dtype=torch.long).unsqueeze(0).to(device)
                    t = torch.tensor([task_id], dtype=torch.long).to(device)
                    pred = model(g, t).squeeze().cpu().numpy()
                h, w = grid.shape
                pred_binary = (pred[:h, :w] > 0.5).astype(float)
                
            elif skill in ['mask_dominant', 'mask_rare']:
                category = 'compositional'
                if skill == 'mask_dominant':
                    pred_binary = compositional_mask_dominant(model, grid, device)
                else:
                    pred_binary = compositional_mask_rare(model, grid, device)
                    
            else:  # algorithmic
                category = 'algorithmic'
                if skill == 'mask_largest_region':
                    pred_binary = algo_mask_largest_region(grid)
                elif skill == 'mask_smallest_region':
                    pred_binary = algo_mask_smallest_region(grid)
                elif skill == 'mask_isolated':
                    pred_binary = algo_mask_isolated(grid)
                elif skill == 'mask_boundary':
                    pred_binary = algo_mask_boundary(grid)
                else:
                    print(f"    ? {skill}: UNKNOWN")
                    continue
            
            # Compare
            match = (pred_binary == expected).all()
            
            if match:
                results[category]['pass'] += 1
                print(f"    ✓ {skill}")
            else:
                results[category]['fail'] += 1
                diff = int((pred_binary != expected).sum())
                print(f"    ✗ {skill} ({diff} pixels wrong)")
                results[category]['details'].append({
                    'grid': name,
                    'skill': skill,
                    'pixels_wrong': diff
                })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for category, data in results.items():
        total = data['pass'] + data['fail']
        if total > 0:
            acc = data['pass'] / total
            print(f"\n{category.upper()}:")
            print(f"  Passed: {data['pass']}/{total} = {acc:.1%}")
            if data['details']:
                print(f"  Failures:")
                for d in data['details']:
                    print(f"    - {d['grid']}/{d['skill']}: {d['pixels_wrong']} pixels")
    
    # Overall
    total_pass = sum(r['pass'] for r in results.values())
    total_fail = sum(r['fail'] for r in results.values())
    total = total_pass + total_fail
    
    print(f"\nOVERALL: {total_pass}/{total} = {total_pass/total:.1%}")
    
    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = load_model(device)
    results = evaluate_comprehensive(model, device)
