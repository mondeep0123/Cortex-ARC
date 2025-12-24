"""
Grid utilities for ARC-AGI tasks.

This module provides core operations for manipulating and analyzing grids.
"""

import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
from collections import Counter

# Type aliases
Grid = np.ndarray  # 2D array of integers (0-9)
Position = Tuple[int, int]  # (row, col)
Color = int  # Integer 0-9
BoundingBox = Tuple[int, int, int, int]  # (row_min, row_max, col_min, col_max)


@dataclass
class GridObject:
    """Represents a connected object in a grid."""
    mask: Grid  # Boolean mask of object location
    positions: Set[Position]  # Set of (row, col) positions
    color: Color  # Primary color of object
    bbox: BoundingBox  # Bounding box (r_min, r_max, c_min, c_max)
    
    @property
    def size(self) -> int:
        """Number of cells in object."""
        return len(self.positions)
    
    @property
    def height(self) -> int:
        """Height of bounding box."""
        return self.bbox[1] - self.bbox[0] + 1
    
    @property
    def width(self) -> int:
        """Width of bounding box."""
        return self.bbox[3] - self.bbox[2] + 1
    
    def extract_subgrid(self, grid: Grid) -> Grid:
        """Extract the object as a tight subgrid."""
        r_min, r_max, c_min, c_max = self.bbox
        return grid[r_min:r_max+1, c_min:c_max+1].copy()


def create_grid(height: int, width: int, fill_color: Color = 0) -> Grid:
    """Create a grid filled with a specific color."""
    return np.full((height, width), fill_color, dtype=np.int32)


def grid_from_list(grid_list: List[List[int]]) -> Grid:
    """Convert a list of lists to a numpy grid."""
    return np.array(grid_list, dtype=np.int32)


def grid_to_list(grid: Grid) -> List[List[int]]:
    """Convert a numpy grid to a list of lists."""
    return grid.tolist()


def get_unique_colors(grid: Grid, exclude_background: bool = True) -> Set[Color]:
    """Get set of unique colors in grid."""
    colors = set(np.unique(grid))
    if exclude_background and 0 in colors:
        colors.remove(0)
    return colors


def count_colors(grid: Grid) -> Dict[Color, int]:
    """Count occurrences of each color."""
    return dict(Counter(grid.flatten()))


def get_background_color(grid: Grid) -> Color:
    """Determine the most common color (assumed to be background)."""
    counts = count_colors(grid)
    return max(counts, key=counts.get)


def find_objects(grid: Grid, background: Optional[Color] = None,
                connectivity: int = 4) -> List[GridObject]:
    """
    Find connected components (objects) in grid.
    
    Args:
        grid: Input grid
        background: Background color to ignore (default: most common color)
        connectivity: 4 or 8-connectivity
    
    Returns:
        List of GridObject instances
    """
    if background is None:
        background = get_background_color(grid)
    
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    objects = []
    
    def flood_fill(start_r: int, start_c: int, target_color: Color) -> GridObject:
        """Flood fill to find connected component."""
        positions = set()
        stack = [(start_r, start_c)]
        r_min, r_max = start_r, start_r
        c_min, c_max = start_c, start_c
        
        while stack:
            r, c = stack.pop()
            
            if r < 0 or r >= height or c < 0 or c >= width:
                continue
            if visited[r, c] or grid[r, c] != target_color:
                continue
            
            visited[r, c] = True
            positions.add((r, c))
            
            # Update bounding box
            r_min, r_max = min(r_min, r), max(r_max, r)
            c_min, c_max = min(c_min, c), max(c_max, c)
            
            # Add neighbors (4-connectivity)
            stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
            
            # Add diagonals for 8-connectivity
            if connectivity == 8:
                stack.extend([(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)])
        
        # Create mask
        mask = np.zeros((height, width), dtype=bool)
        for r, c in positions:
            mask[r, c] = True
        
        return GridObject(
            mask=mask,
            positions=positions,
            color=target_color,
            bbox=(r_min, r_max, c_min, c_max)
        )
    
    # Find all objects
    for r in range(height):
        for c in range(width):
            if not visited[r, c] and grid[r, c] != background:
                obj = flood_fill(r, c, grid[r, c])
                objects.append(obj)
    
    return objects


def apply_transform(grid: Grid, transform: str, **kwargs) -> Grid:
    """
    Apply a transformation to a grid.
    
    Args:
        grid: Input grid
        transform: Transformation name ('rotate_90', 'rotate_180', 'rotate_270',
                   'flip_h', 'flip_v', 'transpose')
        **kwargs: Additional transformation parameters
    
    Returns:
        Transformed grid
    """
    if transform == 'rotate_90':
        return np.rot90(grid, k=1)
    elif transform == 'rotate_180':
        return np.rot90(grid, k=2)
    elif transform == 'rotate_270':
        return np.rot90(grid, k=3)
    elif transform == 'flip_h':
        return np.flip(grid, axis=1)
    elif transform == 'flip_v':
        return np.flip(grid, axis=0)
    elif transform == 'transpose':
        return grid.T
    else:
        raise ValueError(f"Unknown transform: {transform}")


def has_symmetry(grid: Grid, axis: str = 'vertical') -> bool:
    """Check if grid has symmetry along an axis."""
    if axis == 'vertical':
        return np.array_equal(grid, np.flip(grid, axis=0))
    elif axis == 'horizontal':
        return np.array_equal(grid, np.flip(grid, axis=1))
    elif axis == 'diagonal':
        return np.array_equal(grid, grid.T)
    elif axis == 'anti_diagonal':
        return np.array_equal(grid, np.flip(grid.T, axis=1))
    else:
        raise ValueError(f"Unknown axis: {axis}")


def detect_symmetries(grid: Grid) -> List[str]:
    """Detect all symmetries present in grid."""
    symmetries = []
    for axis in ['vertical', 'horizontal', 'diagonal', 'anti_diagonal']:
        if has_symmetry(grid, axis):
            symmetries.append(axis)
    return symmetries


def crop_to_content(grid: Grid, background: Color = 0) -> Tuple[Grid, BoundingBox]:
    """
    Crop grid to smallest bounding box containing all non-background cells.
    
    Returns:
        Cropped grid and original bounding box coordinates
    """
    non_bg = np.argwhere(grid != background)
    
    if len(non_bg) == 0:
        # All background
        return grid, (0, grid.shape[0]-1, 0, grid.shape[1]-1)
    
    r_min, c_min = non_bg.min(axis=0)
    r_max, c_max = non_bg.max(axis=0)
    
    cropped = grid[r_min:r_max+1, c_min:c_max+1].copy()
    return cropped, (r_min, r_max, c_min, c_max)


def pad_grid(grid: Grid, padding: int, fill_color: Color = 0) -> Grid:
    """Add padding around grid."""
    return np.pad(grid, padding, mode='constant', constant_values=fill_color)


def resize_grid(grid: Grid, new_height: int, new_width: int, 
                fill_color: Color = 0, align: str = 'center') -> Grid:
    """
    Resize grid to new dimensions.
    
    Args:
        grid: Input grid
        new_height: Target height
        new_width: Target width
        fill_color: Color for new cells
        align: 'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    """
    result = create_grid(new_height, new_width, fill_color)
    
    old_h, old_w = grid.shape
    
    # Calculate placement based on alignment
    if align == 'center':
        r_offset = (new_height - old_h) // 2
        c_offset = (new_width - old_w) // 2
    elif align == 'top_left':
        r_offset, c_offset = 0, 0
    elif align == 'top_right':
        r_offset, c_offset = 0, new_width - old_w
    elif align == 'bottom_left':
        r_offset, c_offset = new_height - old_h, 0
    elif align == 'bottom_right':
        r_offset, c_offset = new_height - old_h, new_width - old_w
    else:
        raise ValueError(f"Unknown alignment: {align}")
    
    # Ensure we don't go out of bounds
    r_offset = max(0, r_offset)
    c_offset = max(0, c_offset)
    
    # Copy what fits
    copy_h = min(old_h, new_height - r_offset)
    copy_w = min(old_w, new_width - c_offset)
    
    result[r_offset:r_offset+copy_h, c_offset:c_offset+copy_w] = grid[:copy_h, :copy_w]
    
    return result


def tile_grid(grid: Grid, tiles_v: int, tiles_h: int) -> Grid:
    """Tile a grid vertically and horizontally."""
    return np.tile(grid, (tiles_v, tiles_h))


def translate_grid(grid: Grid, offset_r: int, offset_c: int, 
                   background: Color = 0, wrap: bool = False) -> Grid:
    """
    Translate grid content by offset.
    
    Args:
        grid: Input grid
        offset_r: Vertical offset (positive = down)
        offset_c: Horizontal offset (positive = right)
        background: Fill color for empty space
        wrap: If True, wrap around edges; if False, clip
    """
    if wrap:
        return np.roll(grid, (offset_r, offset_c), axis=(0, 1))
    else:
        result = create_grid(grid.shape[0], grid.shape[1], background)
        
        # Calculate source and destination regions
        src_r_start = max(0, -offset_r)
        src_r_end = min(grid.shape[0], grid.shape[0] - offset_r)
        src_c_start = max(0, -offset_c)
        src_c_end = min(grid.shape[1], grid.shape[1] - offset_c)
        
        dst_r_start = max(0, offset_r)
        dst_r_end = dst_r_start + (src_r_end - src_r_start)
        dst_c_start = max(0, offset_c)
        dst_c_end = dst_c_start + (src_c_end - src_c_start)
        
        # Copy
        result[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
            grid[src_r_start:src_r_end, src_c_start:src_c_end]
        
        return result


def apply_color_map(grid: Grid, color_map: Dict[Color, Color]) -> Grid:
    """Apply a color mapping to grid."""
    result = grid.copy()
    for old_color, new_color in color_map.items():
        result[grid == old_color] = new_color
    return result


def overlay_grids(base: Grid, overlay: Grid, position: Position,
                 transparent_color: Optional[Color] = None) -> Grid:
    """
    Overlay one grid on top of another.
    
    Args:
        base: Base grid
        overlay: Grid to overlay
        position: (row, col) position for top-left of overlay
        transparent_color: If specified, this color in overlay won't overwrite base
    """
    result = base.copy()
    r_offset, c_offset = position
    
    for r in range(overlay.shape[0]):
        for c in range(overlay.shape[1]):
            target_r = r_offset + r
            target_c = c_offset + c
            
            # Check bounds
            if (0 <= target_r < base.shape[0] and 
                0 <= target_c < base.shape[1]):
                
                overlay_color = overlay[r, c]
                
                # Apply overlay if not transparent
                if transparent_color is None or overlay_color != transparent_color:
                    result[target_r, target_c] = overlay_color
    
    return result


def grid_equals(grid1: Grid, grid2: Grid) -> bool:
    """Check if two grids are equal."""
    if grid1.shape != grid2.shape:
        return False
    return np.array_equal(grid1, grid2)


def grid_similarity(grid1: Grid, grid2: Grid) -> float:
    """
    Calculate similarity between two grids (0.0 to 1.0).
    Returns fraction of matching cells (considering size differences).
    """
    # Pad to same size
    max_h = max(grid1.shape[0], grid2.shape[0])
    max_w = max(grid1.shape[1], grid2.shape[1])
    
    g1_padded = resize_grid(grid1, max_h, max_w, fill_color=-1, align='top_left')
    g2_padded = resize_grid(grid2, max_h, max_w, fill_color=-1, align='top_left')
    
    matches = np.sum(g1_padded == g2_padded)
    total = max_h * max_w
    
    return matches / total if total > 0 else 0.0
