"""Task visualization utilities."""

from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Optional, List

from ..core.task import Task
from ..core.grid import Grid
from ..solvers.base import SolverResult
from .grid_viz import plot_grid, plot_grids


def plot_task(
    task: Task,
    figsize: Optional[tuple] = None,
    show_test_output: bool = True
) -> plt.Figure:
    """
    Plot all examples in a task.
    
    Shows training pairs and test pairs (with or without outputs).
    """
    n_train = len(task.train)
    n_test = len(task.test)
    n_rows = n_train + n_test
    
    if figsize is None:
        figsize = (8, n_rows * 2)
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot training pairs
    for i, pair in enumerate(task.train):
        plot_grid(pair.input, ax=axes[i, 0], title=f"Train {i+1} Input")
        plot_grid(pair.output, ax=axes[i, 1], title=f"Train {i+1} Output")
    
    # Plot test pairs
    for i, pair in enumerate(task.test):
        row = n_train + i
        plot_grid(pair.input, ax=axes[row, 0], title=f"Test {i+1} Input")
        
        if show_test_output:
            plot_grid(pair.output, ax=axes[row, 1], title=f"Test {i+1} Output")
        else:
            axes[row, 1].text(0.5, 0.5, "?", ha='center', va='center',
                             fontsize=48, transform=axes[row, 1].transAxes)
            axes[row, 1].set_title(f"Test {i+1} Output")
            axes[row, 1].axis('off')
    
    fig.suptitle(f"Task: {task.task_id}", fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_task_comparison(
    task: Task,
    result: SolverResult,
    figsize: Optional[tuple] = None
) -> plt.Figure:
    """
    Plot task with solver predictions compared to ground truth.
    
    Shows training pairs and test predictions.
    """
    n_train = len(task.train)
    n_test = len(task.test)
    n_rows = n_train + n_test
    
    if figsize is None:
        figsize = (12, n_rows * 2)
    
    # 3 columns: input, expected output, prediction
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot training pairs
    for i, pair in enumerate(task.train):
        plot_grid(pair.input, ax=axes[i, 0], title=f"Train {i+1} Input")
        plot_grid(pair.output, ax=axes[i, 1], title=f"Train {i+1} Output")
        axes[i, 2].text(0.5, 0.5, "-", ha='center', va='center',
                       fontsize=24, transform=axes[i, 2].transAxes)
        axes[i, 2].set_title("(training)")
        axes[i, 2].axis('off')
    
    # Plot test pairs with predictions
    for i, (pair, preds) in enumerate(zip(task.test, result.predictions)):
        row = n_train + i
        
        plot_grid(pair.input, ax=axes[row, 0], title=f"Test {i+1} Input")
        plot_grid(pair.output, ax=axes[row, 1], title=f"Test {i+1} Expected")
        
        if preds:
            pred = preds[0]  # Show first prediction
            plot_grid(pred, ax=axes[row, 2])
            
            # Color title based on correctness
            if pred == pair.output:
                axes[row, 2].set_title(f"Prediction ✓", color='green')
            else:
                similarity = pred.similarity(pair.output)
                axes[row, 2].set_title(f"Prediction ({similarity:.0%})", color='red')
        else:
            axes[row, 2].text(0.5, 0.5, "No prediction", ha='center', va='center',
                             fontsize=12, transform=axes[row, 2].transAxes)
            axes[row, 2].set_title("Prediction")
            axes[row, 2].axis('off')
    
    fig.suptitle(f"Task: {task.task_id} | Solver: {result.metadata.get('solver', 'unknown')}", 
                fontsize=14)
    plt.tight_layout()
    
    return fig


def create_task_animation(
    task: Task,
    output_path: str = "task_animation.gif",
    duration: float = 1.0
) -> str:
    """
    Create an animated GIF showing the task.
    
    Cycles through training examples.
    """
    import imageio
    from io import BytesIO
    
    frames = []
    
    for i, pair in enumerate(task.train):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        plot_grid(pair.input, ax=axes[0], title=f"Example {i+1} Input")
        plot_grid(pair.output, ax=axes[1], title=f"Example {i+1} Output")
        fig.suptitle(f"Task: {task.task_id}", fontsize=12)
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)
    
    # Save animation
    imageio.mimsave(output_path, frames, duration=duration)
    
    return output_path
