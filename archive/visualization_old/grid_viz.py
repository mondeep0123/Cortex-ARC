"""Grid visualization utilities."""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import numpy as np
from typing import List, Optional, Tuple

from ..core.grid import Grid, ARC_COLORS


# Create ARC colormap
def get_arc_cmap():
    """Get matplotlib colormap for ARC colors."""
    arc_palette = [
        tuple(c / 255 for c in ARC_COLORS[i]) for i in range(10)
    ]
    return colors.ListedColormap(arc_palette)


ARC_CMAP = get_arc_cmap()


def plot_grid(
    grid: Grid,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_values: bool = False,
    cell_size: float = 0.5,
    show_grid_lines: bool = True
) -> plt.Axes:
    """
    Plot a single grid.
    
    Args:
        grid: Grid to plot
        ax: Matplotlib axes (creates new figure if None)
        title: Title for the plot
        show_values: Show numeric values in cells
        cell_size: Size of each cell in inches
        show_grid_lines: Whether to show grid lines
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(grid.width * cell_size + 0.5, grid.height * cell_size + 0.5)
        )
    
    # Plot the grid
    im = ax.imshow(
        grid.data,
        cmap=ARC_CMAP,
        vmin=0,
        vmax=9,
        aspect='equal'
    )
    
    # Add grid lines
    if show_grid_lines:
        ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
    
    # Show values
    if show_values:
        for i in range(grid.height):
            for j in range(grid.width):
                val = grid.data[i, j]
                # Choose text color based on background
                text_color = 'white' if val in [0, 1, 6, 9] else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                       color=text_color, fontsize=8, fontweight='bold')
    
    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    if title:
        ax.set_title(title, fontsize=10)
    
    return ax


def plot_grids(
    grids: List[Grid],
    titles: Optional[List[str]] = None,
    ncols: int = 4,
    cell_size: float = 0.4,
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple grids in a grid layout.
    
    Args:
        grids: List of grids to plot
        titles: Optional titles for each grid
        ncols: Number of columns
        cell_size: Size of each cell
        figsize: Figure size (auto-calculated if None)
        suptitle: Super title for the figure
    
    Returns:
        Matplotlib figure
    """
    n = len(grids)
    nrows = (n + ncols - 1) // ncols
    
    if figsize is None:
        max_w = max(g.width for g in grids)
        max_h = max(g.height for g in grids)
        figsize = (ncols * max_w * cell_size + 1, nrows * max_h * cell_size + 1)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, grid in enumerate(grids):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        title = titles[idx] if titles and idx < len(titles) else None
        plot_grid(grid, ax=ax, title=title, cell_size=cell_size)
    
    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_comparison(
    input_grid: Grid,
    ground_truth: Grid,
    prediction: Grid,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot input, ground truth, and prediction side by side.
    
    Useful for visualizing solver output.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    plot_grid(input_grid, ax=axes[0], title="Input")
    plot_grid(ground_truth, ax=axes[1], title="Expected")
    plot_grid(prediction, ax=axes[2], title="Prediction")
    
    # Highlight differences
    if prediction == ground_truth:
        axes[2].set_title("Prediction ✓", color='green')
    else:
        axes[2].set_title("Prediction ✗", color='red')
    
    if title:
        fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    return fig
