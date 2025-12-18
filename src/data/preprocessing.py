"""Preprocessing utilities for ARC-AGI grids."""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..core.grid import Grid
from ..core.task import Task, TaskPair


def normalize_colors(grid: Grid) -> Tuple[Grid, Dict[int, int]]:
    """
    Normalize colors to use consecutive integers starting from 0.
    
    This can help with learning by making color indices consistent.
    
    Args:
        grid: Grid to normalize
    
    Returns:
        Tuple of (normalized grid, mapping from new to original colors)
    """
    unique_colors = sorted(grid.unique_colors())
    
    # Create mapping
    color_map = {old: new for new, old in enumerate(unique_colors)}
    reverse_map = {new: old for old, new in color_map.items()}
    
    # Apply mapping
    result = grid.data.copy()
    for old, new in color_map.items():
        result[grid.data == old] = new
    
    return Grid(data=result), reverse_map


def denormalize_colors(grid: Grid, reverse_map: Dict[int, int]) -> Grid:
    """Reverse the color normalization."""
    result = grid.data.copy()
    for new, old in reverse_map.items():
        result[grid.data == new] = old
    return Grid(data=result)


def preprocess_grid(
    grid: Grid,
    normalize: bool = False,
    pad_to_size: Optional[Tuple[int, int]] = None,
    center: bool = False
) -> Grid:
    """
    Preprocess a grid for model input.
    
    Args:
        grid: Grid to preprocess
        normalize: Normalize color indices
        pad_to_size: Pad to fixed size (height, width)
        center: Center content when padding
    
    Returns:
        Preprocessed grid
    """
    result = grid.copy()
    
    if normalize:
        result, _ = normalize_colors(result)
    
    if pad_to_size is not None:
        target_h, target_w = pad_to_size
        
        if center:
            # Center the content
            pad_h = target_h - result.height
            pad_w = target_w - result.width
            
            top = pad_h // 2
            left = pad_w // 2
            
            padded = np.zeros((target_h, target_w), dtype=np.int8)
            padded[top:top+result.height, left:left+result.width] = result.data
            result = Grid(data=padded)
        else:
            # Pad bottom-right
            result = result.resize(target_h, target_w, fill_color=0)
    
    return result


def grid_to_tensor(grid: Grid, one_hot: bool = False) -> np.ndarray:
    """
    Convert grid to tensor format for neural networks.
    
    Args:
        grid: Grid to convert
        one_hot: If True, return one-hot encoded tensor (H, W, 10)
                 If False, return integer tensor (H, W)
    
    Returns:
        Numpy array suitable for model input
    """
    if one_hot:
        tensor = np.zeros((grid.height, grid.width, 10), dtype=np.float32)
        for i in range(10):
            tensor[:, :, i] = (grid.data == i).astype(np.float32)
        return tensor
    else:
        return grid.data.astype(np.int64)


def tensor_to_grid(tensor: np.ndarray) -> Grid:
    """
    Convert tensor back to grid.
    
    Args:
        tensor: Either (H, W) integer tensor or (H, W, 10) one-hot tensor
    
    Returns:
        Grid
    """
    if tensor.ndim == 3:
        # One-hot encoded - take argmax
        data = np.argmax(tensor, axis=-1).astype(np.int8)
    else:
        data = tensor.astype(np.int8)
    
    return Grid(data=data)


def extract_objects(grid: Grid, bg_color: int = 0) -> List[Tuple[Grid, Tuple[int, int]]]:
    """
    Extract connected objects from a grid.
    
    Args:
        grid: Grid to extract objects from
        bg_color: Background color to ignore
    
    Returns:
        List of (object_grid, (row_offset, col_offset)) tuples
    """
    from scipy import ndimage
    
    # Create binary mask of non-background
    mask = (grid.data != bg_color).astype(np.int8)
    
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    objects = []
    for i in range(1, num_features + 1):
        # Find bounding box of this object
        positions = np.where(labeled == i)
        min_row, max_row = positions[0].min(), positions[0].max()
        min_col, max_col = positions[1].min(), positions[1].max()
        
        # Extract the object
        obj_data = np.where(
            labeled[min_row:max_row+1, min_col:max_col+1] == i,
            grid.data[min_row:max_row+1, min_col:max_col+1],
            bg_color
        ).astype(np.int8)
        
        objects.append((Grid(data=obj_data), (min_row, min_col)))
    
    return objects


def find_patterns(grid: Grid, pattern_size: Tuple[int, int]) -> List[Tuple[Grid, Tuple[int, int]]]:
    """
    Find all unique patterns of a given size in the grid.
    
    Args:
        grid: Grid to search
        pattern_size: (height, width) of patterns to find
    
    Returns:
        List of (pattern_grid, (row, col)) tuples for first occurrence
    """
    ph, pw = pattern_size
    
    if ph > grid.height or pw > grid.width:
        return []
    
    seen_patterns = {}
    
    for row in range(grid.height - ph + 1):
        for col in range(grid.width - pw + 1):
            pattern = grid.data[row:row+ph, col:col+pw]
            pattern_tuple = tuple(pattern.flatten())
            
            if pattern_tuple not in seen_patterns:
                seen_patterns[pattern_tuple] = (Grid(data=pattern.copy()), (row, col))
    
    return list(seen_patterns.values())


def detect_periodicity(grid: Grid, axis: int = 0) -> Optional[int]:
    """
    Detect if the grid has periodic structure along an axis.
    
    Args:
        grid: Grid to analyze
        axis: 0 for vertical periodicity, 1 for horizontal
    
    Returns:
        Period length if periodic, None otherwise
    """
    if axis == 0:
        data = grid.data
        size = grid.height
    else:
        data = grid.data.T
        size = grid.width
    
    # Check possible periods
    for period in range(1, size // 2 + 1):
        if size % period == 0:
            is_periodic = True
            for i in range(period, size):
                if not np.array_equal(data[i], data[i % period]):
                    is_periodic = False
                    break
            if is_periodic:
                return period
    
    return None
