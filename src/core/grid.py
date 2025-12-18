"""Grid representation for ARC-AGI tasks."""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import copy


# ARC color palette (RGB values for visualization)
ARC_COLORS = {
    0: (0, 0, 0),       # Black
    1: (0, 116, 217),   # Blue
    2: (255, 65, 54),   # Red
    3: (46, 204, 64),   # Green
    4: (255, 220, 0),   # Yellow
    5: (170, 170, 170), # Grey
    6: (240, 18, 190),  # Magenta
    7: (255, 133, 27),  # Orange
    8: (127, 219, 255), # Cyan
    9: (135, 12, 37),   # Maroon
}

COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "maroon",
}


@dataclass
class Grid:
    """
    Represents a 2D grid in the ARC-AGI format.
    
    Grids are 2D arrays of integers from 0-9, where each integer
    represents a color. Dimensions can range from 1x1 to 30x30.
    """
    
    data: np.ndarray
    
    def __post_init__(self):
        """Validate grid data after initialization."""
        if isinstance(self.data, list):
            self.data = np.array(self.data, dtype=np.int8)
        
        assert self.data.ndim == 2, f"Grid must be 2D, got {self.data.ndim}D"
        assert self.data.min() >= 0, f"Grid values must be >= 0, got {self.data.min()}"
        assert self.data.max() <= 9, f"Grid values must be <= 9, got {self.data.max()}"
        assert 1 <= self.height <= 30, f"Height must be 1-30, got {self.height}"
        assert 1 <= self.width <= 30, f"Width must be 1-30, got {self.width}"
    
    @classmethod
    def from_list(cls, data: List[List[int]]) -> Grid:
        """Create a Grid from a nested list."""
        return cls(data=np.array(data, dtype=np.int8))
    
    @classmethod
    def zeros(cls, height: int, width: int) -> Grid:
        """Create a grid filled with zeros (black)."""
        return cls(data=np.zeros((height, width), dtype=np.int8))
    
    @classmethod
    def ones(cls, height: int, width: int, color: int = 1) -> Grid:
        """Create a grid filled with a specified color."""
        return cls(data=np.full((height, width), color, dtype=np.int8))
    
    @property
    def height(self) -> int:
        """Return the height of the grid."""
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        """Return the width of the grid."""
        return self.data.shape[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the grid as (height, width)."""
        return self.data.shape
    
    @property
    def size(self) -> int:
        """Return the total number of cells."""
        return self.data.size
    
    def __eq__(self, other: Grid) -> bool:
        """Check if two grids are exactly equal."""
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.data, other.data)
    
    def __hash__(self) -> int:
        """Hash the grid for use in sets/dicts."""
        return hash(self.data.tobytes())
    
    def __getitem__(self, key) -> Union[int, np.ndarray]:
        """Allow indexing into the grid."""
        return self.data[key]
    
    def __setitem__(self, key, value):
        """Allow setting values in the grid."""
        self.data[key] = value
    
    def copy(self) -> Grid:
        """Return a deep copy of the grid."""
        return Grid(data=self.data.copy())
    
    def to_list(self) -> List[List[int]]:
        """Convert to nested Python list."""
        return self.data.tolist()
    
    def to_tuple(self) -> Tuple[Tuple[int, ...], ...]:
        """Convert to nested tuple (hashable)."""
        return tuple(tuple(row) for row in self.data)
    
    # =========== Analysis Methods ===========
    
    def unique_colors(self) -> List[int]:
        """Return sorted list of unique colors in the grid."""
        return sorted(np.unique(self.data).tolist())
    
    def color_counts(self) -> dict:
        """Return a dictionary of color -> count."""
        unique, counts = np.unique(self.data, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def background_color(self) -> int:
        """Estimate the background color (most common color)."""
        counts = self.color_counts()
        return max(counts, key=counts.get)
    
    def find_color(self, color: int) -> List[Tuple[int, int]]:
        """Find all positions of a given color."""
        positions = np.where(self.data == color)
        return list(zip(positions[0].tolist(), positions[1].tolist()))
    
    def bounding_box(self, exclude_color: int = 0) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the bounding box of non-background pixels.
        Returns (min_row, min_col, max_row, max_col) or None if empty.
        """
        mask = self.data != exclude_color
        if not mask.any():
            return None
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]
        
        return (min_row, min_col, max_row, max_col)
    
    def crop(self, min_row: int, min_col: int, max_row: int, max_col: int) -> Grid:
        """Crop the grid to the specified region (inclusive)."""
        return Grid(data=self.data[min_row:max_row+1, min_col:max_col+1].copy())
    
    def crop_to_content(self, exclude_color: int = 0) -> Optional[Grid]:
        """Crop to the bounding box of non-background content."""
        bbox = self.bounding_box(exclude_color)
        if bbox is None:
            return None
        return self.crop(*bbox)
    
    # =========== Transformation Methods ===========
    
    def rotate_90(self) -> Grid:
        """Rotate the grid 90 degrees clockwise."""
        return Grid(data=np.rot90(self.data, k=-1))
    
    def rotate_180(self) -> Grid:
        """Rotate the grid 180 degrees."""
        return Grid(data=np.rot90(self.data, k=2))
    
    def rotate_270(self) -> Grid:
        """Rotate the grid 270 degrees clockwise (90 counter-clockwise)."""
        return Grid(data=np.rot90(self.data, k=1))
    
    def flip_horizontal(self) -> Grid:
        """Flip the grid horizontally (left-right)."""
        return Grid(data=np.fliplr(self.data))
    
    def flip_vertical(self) -> Grid:
        """Flip the grid vertically (up-down)."""
        return Grid(data=np.flipud(self.data))
    
    def transpose(self) -> Grid:
        """Transpose the grid (swap rows and columns)."""
        return Grid(data=self.data.T)
    
    def replace_color(self, old_color: int, new_color: int) -> Grid:
        """Replace all occurrences of old_color with new_color."""
        new_data = self.data.copy()
        new_data[new_data == old_color] = new_color
        return Grid(data=new_data)
    
    def resize(self, new_height: int, new_width: int, fill_color: int = 0) -> Grid:
        """Resize the grid, padding or cropping as needed."""
        new_data = np.full((new_height, new_width), fill_color, dtype=np.int8)
        h = min(self.height, new_height)
        w = min(self.width, new_width)
        new_data[:h, :w] = self.data[:h, :w]
        return Grid(data=new_data)
    
    def tile(self, h_repeat: int, w_repeat: int) -> Grid:
        """Tile the grid h_repeat times vertically and w_repeat times horizontally."""
        return Grid(data=np.tile(self.data, (h_repeat, w_repeat)))
    
    def scale(self, factor: int) -> Grid:
        """Scale up the grid by an integer factor."""
        return Grid(data=np.repeat(np.repeat(self.data, factor, axis=0), factor, axis=1))
    
    # =========== Comparison Methods ===========
    
    def diff(self, other: Grid) -> Optional[Grid]:
        """
        Return a grid showing differences (non-zero where different).
        Returns None if shapes don't match.
        """
        if self.shape != other.shape:
            return None
        diff_mask = (self.data != other.data).astype(np.int8)
        return Grid(data=diff_mask)
    
    def similarity(self, other: Grid) -> float:
        """
        Calculate similarity as fraction of matching cells.
        Returns 0 if shapes don't match.
        """
        if self.shape != other.shape:
            return 0.0
        return np.mean(self.data == other.data)
    
    def __repr__(self) -> str:
        """String representation of the grid."""
        return f"Grid(shape={self.shape}, colors={self.unique_colors()})"
    
    def __str__(self) -> str:
        """Pretty print the grid."""
        lines = []
        for row in self.data:
            lines.append(" ".join(str(x) for x in row))
        return "\n".join(lines)
