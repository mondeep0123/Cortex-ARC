"""DSL Primitives for ARC-AGI program synthesis."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .grid import Grid


class PrimitiveType(Enum):
    """Types of primitives in the DSL."""
    SELECTOR = "selector"      # Select parts of grid
    TRANSFORM = "transform"    # Transform grid
    COMBINER = "combiner"      # Combine grids
    GENERATOR = "generator"    # Generate new grids
    ANALYZER = "analyzer"      # Analyze grid properties


@dataclass
class Primitive(ABC):
    """
    Base class for DSL primitives.
    
    Primitives are the building blocks for program synthesis.
    They can be composed to form complex transformations.
    """
    
    name: str
    ptype: PrimitiveType
    arity: int  # Number of arguments
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Execute the primitive."""
        pass
    
    @abstractmethod
    def signature(self) -> str:
        """Return the type signature of the primitive."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.name}({self.signature()})"


# ============ Selector Primitives ============

class SelectColor(Primitive):
    """Select cells of a specific color."""
    
    def __init__(self):
        super().__init__(
            name="select_color",
            ptype=PrimitiveType.SELECTOR,
            arity=2
        )
    
    def __call__(self, grid: Grid, color: int) -> Grid:
        """Return a mask grid with 1s where color matches."""
        mask = (grid.data == color).astype(np.int8)
        return Grid(data=mask)
    
    def signature(self) -> str:
        return "Grid × Color → Grid"


class SelectNonBackground(Primitive):
    """Select all non-background cells."""
    
    def __init__(self):
        super().__init__(
            name="select_nonbg",
            ptype=PrimitiveType.SELECTOR,
            arity=1
        )
    
    def __call__(self, grid: Grid, bg_color: int = 0) -> Grid:
        """Return a mask grid with 1s for non-background."""
        mask = (grid.data != bg_color).astype(np.int8)
        return Grid(data=mask)
    
    def signature(self) -> str:
        return "Grid → Grid"


class SelectBoundingBox(Primitive):
    """Select the bounding box of non-background content."""
    
    def __init__(self):
        super().__init__(
            name="select_bbox",
            ptype=PrimitiveType.SELECTOR,
            arity=1
        )
    
    def __call__(self, grid: Grid, bg_color: int = 0) -> Optional[Grid]:
        """Return the cropped content."""
        return grid.crop_to_content(exclude_color=bg_color)
    
    def signature(self) -> str:
        return "Grid → Grid"


# ============ Transform Primitives ============

class Rotate(Primitive):
    """Rotate the grid."""
    
    def __init__(self):
        super().__init__(
            name="rotate",
            ptype=PrimitiveType.TRANSFORM,
            arity=2
        )
    
    def __call__(self, grid: Grid, times: int = 1) -> Grid:
        """Rotate 90 degrees clockwise, times number of times."""
        times = times % 4
        if times == 0:
            return grid.copy()
        elif times == 1:
            return grid.rotate_90()
        elif times == 2:
            return grid.rotate_180()
        else:
            return grid.rotate_270()
    
    def signature(self) -> str:
        return "Grid × Int → Grid"


class Flip(Primitive):
    """Flip the grid horizontally or vertically."""
    
    def __init__(self):
        super().__init__(
            name="flip",
            ptype=PrimitiveType.TRANSFORM,
            arity=2
        )
    
    def __call__(self, grid: Grid, axis: str = "horizontal") -> Grid:
        """Flip along specified axis."""
        if axis == "horizontal" or axis == "h":
            return grid.flip_horizontal()
        else:
            return grid.flip_vertical()
    
    def signature(self) -> str:
        return "Grid × Axis → Grid"


class Scale(Primitive):
    """Scale the grid by an integer factor."""
    
    def __init__(self):
        super().__init__(
            name="scale",
            ptype=PrimitiveType.TRANSFORM,
            arity=2
        )
    
    def __call__(self, grid: Grid, factor: int) -> Grid:
        """Scale up by factor."""
        return grid.scale(factor)
    
    def signature(self) -> str:
        return "Grid × Int → Grid"


class ReplaceColor(Primitive):
    """Replace one color with another."""
    
    def __init__(self):
        super().__init__(
            name="replace_color",
            ptype=PrimitiveType.TRANSFORM,
            arity=3
        )
    
    def __call__(self, grid: Grid, old_color: int, new_color: int) -> Grid:
        """Replace old_color with new_color."""
        return grid.replace_color(old_color, new_color)
    
    def signature(self) -> str:
        return "Grid × Color × Color → Grid"


class Tile(Primitive):
    """Tile the grid."""
    
    def __init__(self):
        super().__init__(
            name="tile",
            ptype=PrimitiveType.TRANSFORM,
            arity=3
        )
    
    def __call__(self, grid: Grid, h_repeat: int, w_repeat: int) -> Grid:
        """Tile the grid."""
        return grid.tile(h_repeat, w_repeat)
    
    def signature(self) -> str:
        return "Grid × Int × Int → Grid"


# ============ Combiner Primitives ============

class Overlay(Primitive):
    """Overlay one grid on another."""
    
    def __init__(self):
        super().__init__(
            name="overlay",
            ptype=PrimitiveType.COMBINER,
            arity=3
        )
    
    def __call__(self, base: Grid, overlay: Grid, transparent: int = 0) -> Grid:
        """Overlay grid on base, treating transparent color as see-through."""
        if base.shape != overlay.shape:
            raise ValueError("Grids must have the same shape for overlay")
        
        result = base.data.copy()
        mask = overlay.data != transparent
        result[mask] = overlay.data[mask]
        return Grid(data=result)
    
    def signature(self) -> str:
        return "Grid × Grid × Color → Grid"


class Concat(Primitive):
    """Concatenate two grids."""
    
    def __init__(self):
        super().__init__(
            name="concat",
            ptype=PrimitiveType.COMBINER,
            arity=3
        )
    
    def __call__(self, grid1: Grid, grid2: Grid, axis: int = 0) -> Grid:
        """Concatenate grids along axis (0=vertical, 1=horizontal)."""
        result = np.concatenate([grid1.data, grid2.data], axis=axis)
        return Grid(data=result)
    
    def signature(self) -> str:
        return "Grid × Grid × Axis → Grid"


# ============ Generator Primitives ============

class MakeGrid(Primitive):
    """Create a new grid filled with a color."""
    
    def __init__(self):
        super().__init__(
            name="make_grid",
            ptype=PrimitiveType.GENERATOR,
            arity=3
        )
    
    def __call__(self, height: int, width: int, color: int = 0) -> Grid:
        """Create a grid of specified size and color."""
        return Grid.ones(height, width, color)
    
    def signature(self) -> str:
        return "Int × Int × Color → Grid"


# ============ Primitive Library ============

class PrimitiveLibrary:
    """
    Collection of all available primitives.
    
    This is the DSL (Domain-Specific Language) for ARC-AGI.
    """
    
    def __init__(self):
        self.primitives: Dict[str, Primitive] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register all default primitives."""
        defaults = [
            # Selectors
            SelectColor(),
            SelectNonBackground(),
            SelectBoundingBox(),
            # Transforms
            Rotate(),
            Flip(),
            Scale(),
            ReplaceColor(),
            Tile(),
            # Combiners
            Overlay(),
            Concat(),
            # Generators
            MakeGrid(),
        ]
        
        for p in defaults:
            self.register(p)
    
    def register(self, primitive: Primitive):
        """Register a primitive."""
        self.primitives[primitive.name] = primitive
    
    def get(self, name: str) -> Optional[Primitive]:
        """Get a primitive by name."""
        return self.primitives.get(name)
    
    def __getitem__(self, name: str) -> Primitive:
        """Get a primitive by name."""
        return self.primitives[name]
    
    def list_by_type(self, ptype: PrimitiveType) -> List[Primitive]:
        """List all primitives of a given type."""
        return [p for p in self.primitives.values() if p.ptype == ptype]
    
    def all(self) -> List[Primitive]:
        """Get all primitives."""
        return list(self.primitives.values())
    
    def __repr__(self) -> str:
        return f"PrimitiveLibrary({len(self.primitives)} primitives)"
    
    def summary(self) -> str:
        """Generate a summary of all primitives."""
        lines = ["Primitive Library:"]
        for ptype in PrimitiveType:
            prims = self.list_by_type(ptype)
            if prims:
                lines.append(f"\n{ptype.value.upper()}:")
                for p in prims:
                    lines.append(f"  {p.name}: {p.signature()}")
        return "\n".join(lines)
