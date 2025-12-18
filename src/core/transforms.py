"""Transform functions for ARC-AGI grids."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import ndimage

from .grid import Grid


@dataclass
class Transform(ABC):
    """Base class for grid transformations."""
    
    name: str
    
    @abstractmethod
    def __call__(self, grid: Grid, **kwargs) -> Grid:
        """Apply the transform to a grid."""
        pass
    
    def __repr__(self) -> str:
        return f"Transform({self.name})"


# ============ Geometric Transforms ============

class RotateTransform(Transform):
    """Rotate grid by 90-degree increments."""
    
    def __init__(self):
        super().__init__(name="rotate")
    
    def __call__(self, grid: Grid, k: int = 1) -> Grid:
        """Rotate k*90 degrees clockwise."""
        return Grid(data=np.rot90(grid.data, k=-k))


class FlipTransform(Transform):
    """Flip grid along an axis."""
    
    def __init__(self):
        super().__init__(name="flip")
    
    def __call__(self, grid: Grid, axis: int = 0) -> Grid:
        """Flip along axis (0=vertical, 1=horizontal)."""
        if axis == 0:
            return Grid(data=np.flipud(grid.data))
        else:
            return Grid(data=np.fliplr(grid.data))


class TransposeTransform(Transform):
    """Transpose the grid."""
    
    def __init__(self):
        super().__init__(name="transpose")
    
    def __call__(self, grid: Grid) -> Grid:
        return Grid(data=grid.data.T)


class ScaleTransform(Transform):
    """Scale grid by integer factor."""
    
    def __init__(self):
        super().__init__(name="scale")
    
    def __call__(self, grid: Grid, factor: int = 2) -> Grid:
        return grid.scale(factor)


class CropTransform(Transform):
    """Crop grid to region."""
    
    def __init__(self):
        super().__init__(name="crop")
    
    def __call__(self, grid: Grid, 
                 top: int = 0, left: int = 0, 
                 bottom: Optional[int] = None, 
                 right: Optional[int] = None) -> Grid:
        bottom = bottom or grid.height
        right = right or grid.width
        return Grid(data=grid.data[top:bottom, left:right].copy())


class PadTransform(Transform):
    """Pad grid with a color."""
    
    def __init__(self):
        super().__init__(name="pad")
    
    def __call__(self, grid: Grid, 
                 pad_width: int = 1, 
                 color: int = 0) -> Grid:
        padded = np.pad(
            grid.data, 
            pad_width=pad_width, 
            mode='constant', 
            constant_values=color
        )
        return Grid(data=padded)


# ============ Color Transforms ============

class ColorMapTransform(Transform):
    """Map colors to other colors."""
    
    def __init__(self):
        super().__init__(name="color_map")
    
    def __call__(self, grid: Grid, mapping: Dict[int, int]) -> Grid:
        """Apply color mapping."""
        result = grid.data.copy()
        for old_color, new_color in mapping.items():
            result[grid.data == old_color] = new_color
        return Grid(data=result)


class InvertColorsTransform(Transform):
    """Invert colors (9 - color)."""
    
    def __init__(self):
        super().__init__(name="invert_colors")
    
    def __call__(self, grid: Grid) -> Grid:
        return Grid(data=9 - grid.data)


class SwapColorsTransform(Transform):
    """Swap two colors."""
    
    def __init__(self):
        super().__init__(name="swap_colors")
    
    def __call__(self, grid: Grid, color1: int, color2: int) -> Grid:
        result = grid.data.copy()
        mask1 = grid.data == color1
        mask2 = grid.data == color2
        result[mask1] = color2
        result[mask2] = color1
        return Grid(data=result)


# ============ Morphological Transforms ============

class DilateTransform(Transform):
    """Dilate non-zero regions."""
    
    def __init__(self):
        super().__init__(name="dilate")
    
    def __call__(self, grid: Grid, iterations: int = 1, color: Optional[int] = None) -> Grid:
        """Dilate the specified color (or all non-zero if None)."""
        if color is not None:
            mask = (grid.data == color).astype(np.int8)
        else:
            mask = (grid.data != 0).astype(np.int8)
        
        dilated = ndimage.binary_dilation(mask, iterations=iterations)
        
        result = grid.data.copy()
        new_pixels = dilated & (mask == 0)
        if color is not None:
            result[new_pixels] = color
        else:
            # Use the most common non-zero color
            colors, counts = np.unique(grid.data[grid.data != 0], return_counts=True)
            if len(colors) > 0:
                dominant = colors[np.argmax(counts)]
                result[new_pixels] = dominant
        
        return Grid(data=result)


class ErodeTransform(Transform):
    """Erode non-zero regions."""
    
    def __init__(self):
        super().__init__(name="erode")
    
    def __call__(self, grid: Grid, iterations: int = 1, color: Optional[int] = None) -> Grid:
        """Erode the specified color (or all non-zero if None)."""
        if color is not None:
            mask = (grid.data == color).astype(np.int8)
        else:
            mask = (grid.data != 0).astype(np.int8)
        
        eroded = ndimage.binary_erosion(mask, iterations=iterations)
        
        result = grid.data.copy()
        removed_pixels = mask.astype(bool) & ~eroded
        result[removed_pixels] = 0
        
        return Grid(data=result)


class FillHolesTransform(Transform):
    """Fill holes in regions."""
    
    def __init__(self):
        super().__init__(name="fill_holes")
    
    def __call__(self, grid: Grid, color: int) -> Grid:
        """Fill holes in regions of the specified color."""
        mask = (grid.data == color).astype(np.int8)
        filled = ndimage.binary_fill_holes(mask)
        
        result = grid.data.copy()
        result[filled] = color
        
        return Grid(data=result)


# ============ Pattern Transforms ============

class TileTransform(Transform):
    """Tile the grid."""
    
    def __init__(self):
        super().__init__(name="tile")
    
    def __call__(self, grid: Grid, h_tiles: int = 2, w_tiles: int = 2) -> Grid:
        return grid.tile(h_tiles, w_tiles)


class MirrorTransform(Transform):
    """Create a mirrored version (original + flipped)."""
    
    def __init__(self):
        super().__init__(name="mirror")
    
    def __call__(self, grid: Grid, axis: int = 1) -> Grid:
        """Mirror along axis (0=vertical creates top+bottom, 1=horizontal creates left+right)."""
        if axis == 1:
            flipped = np.fliplr(grid.data)
            result = np.concatenate([grid.data, flipped], axis=1)
        else:
            flipped = np.flipud(grid.data)
            result = np.concatenate([grid.data, flipped], axis=0)
        return Grid(data=result)


class GravityTransform(Transform):
    """Apply gravity - move non-zero cells in a direction."""
    
    def __init__(self):
        super().__init__(name="gravity")
    
    def __call__(self, grid: Grid, direction: str = "down", bg_color: int = 0) -> Grid:
        """Apply gravity in the specified direction (up, down, left, right)."""
        result = np.full_like(grid.data, bg_color)
        
        if direction == "down":
            for col in range(grid.width):
                non_bg = grid.data[:, col][grid.data[:, col] != bg_color]
                if len(non_bg) > 0:
                    result[-len(non_bg):, col] = non_bg
        
        elif direction == "up":
            for col in range(grid.width):
                non_bg = grid.data[:, col][grid.data[:, col] != bg_color]
                if len(non_bg) > 0:
                    result[:len(non_bg), col] = non_bg
        
        elif direction == "right":
            for row in range(grid.height):
                non_bg = grid.data[row, :][grid.data[row, :] != bg_color]
                if len(non_bg) > 0:
                    result[row, -len(non_bg):] = non_bg
        
        elif direction == "left":
            for row in range(grid.height):
                non_bg = grid.data[row, :][grid.data[row, :] != bg_color]
                if len(non_bg) > 0:
                    result[row, :len(non_bg)] = non_bg
        
        return Grid(data=result)


# ============ Transform Library ============

class TransformLibrary:
    """Collection of all available transforms."""
    
    def __init__(self):
        self.transforms: Dict[str, Transform] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register all default transforms."""
        defaults = [
            # Geometric
            RotateTransform(),
            FlipTransform(),
            TransposeTransform(),
            ScaleTransform(),
            CropTransform(),
            PadTransform(),
            # Color
            ColorMapTransform(),
            InvertColorsTransform(),
            SwapColorsTransform(),
            # Morphological
            DilateTransform(),
            ErodeTransform(),
            FillHolesTransform(),
            # Pattern
            TileTransform(),
            MirrorTransform(),
            GravityTransform(),
        ]
        
        for t in defaults:
            self.register(t)
    
    def register(self, transform: Transform):
        """Register a transform."""
        self.transforms[transform.name] = transform
    
    def get(self, name: str) -> Optional[Transform]:
        """Get a transform by name."""
        return self.transforms.get(name)
    
    def __getitem__(self, name: str) -> Transform:
        return self.transforms[name]
    
    def all(self) -> List[Transform]:
        """Get all transforms."""
        return list(self.transforms.values())
    
    def __repr__(self) -> str:
        return f"TransformLibrary({len(self.transforms)} transforms)"
