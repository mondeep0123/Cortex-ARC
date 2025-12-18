"""Analysis and comparison visualization."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional

from ..data.loader import ARCDataset
from ..evaluation.evaluator import EvaluationRun


def plot_task_distribution(
    dataset: ARCDataset,
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot distribution of task characteristics.
    
    Shows:
    - Grid size distribution
    - Number of colors
    - Number of training examples
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Collect statistics
    input_sizes = []
    output_sizes = []
    num_colors = []
    num_train = []
    
    for task in dataset:
        for pair in task.train:
            input_sizes.append(pair.input.size)
            output_sizes.append(pair.output.size)
        num_colors.append(len(task.all_colors()))
        num_train.append(task.num_train)
    
    # Plot input sizes
    axes[0, 0].hist(input_sizes, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Input Grid Size (cells)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Input Size Distribution')
    
    # Plot output sizes
    axes[0, 1].hist(output_sizes, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Output Grid Size (cells)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Output Size Distribution')
    
    # Plot number of colors
    axes[1, 0].hist(num_colors, bins=range(1, 12), edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Number of Colors')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Colors Per Task')
    axes[1, 0].set_xticks(range(1, 11))
    
    # Plot number of training examples
    axes[1, 1].hist(num_train, bins=range(1, 12), edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Number of Training Examples')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Training Examples Per Task')
    axes[1, 1].set_xticks(range(1, 11))
    
    fig.suptitle(f"Dataset: {dataset.version}/{dataset.split} ({len(dataset)} tasks)", fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_solver_comparison(
    runs: Dict[str, EvaluationRun],
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot comparison of multiple solvers.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    names = list(runs.keys())
    accuracies = [runs[n].accuracy for n in names]
    partial_accs = [runs[n].partial_accuracy for n in names]
    times = [runs[n].total_time for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    # Accuracy comparison
    axes[0].bar(x - width/2, accuracies, width, label='Exact Match', color='#2ecc71')
    axes[0].bar(x + width/2, partial_accs, width, label='Partial Match', color='#3498db')
    axes[0].set_xlabel('Solver')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    
    # Time comparison
    axes[1].bar(names, times, color='#e74c3c')
    axes[1].set_xlabel('Solver')
    axes[1].set_ylabel('Total Time (s)')
    axes[1].set_title('Runtime Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def plot_difficulty_analysis(
    run: EvaluationRun,
    dataset: ARCDataset,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Analyze solver performance by task difficulty.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    task_dict = {t.task_id: t for t in dataset}
    
    difficulties = []
    accuracies = []
    num_colors_list = []
    
    for task_id, result in run.task_results.items():
        if task_id in task_dict:
            task = task_dict[task_id]
            difficulties.append(task.difficulty_estimate())
            accuracies.append(result.get('accuracy', 0))
            num_colors_list.append(len(task.all_colors()))
    
    # Difficulty vs accuracy
    axes[0].scatter(difficulties, accuracies, alpha=0.5)
    axes[0].set_xlabel('Estimated Difficulty')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Task Difficulty')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-0.1, 1.1)
    
    # Colors vs accuracy
    axes[1].scatter(num_colors_list, accuracies, alpha=0.5, color='orange')
    axes[1].set_xlabel('Number of Colors')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy vs Number of Colors')
    axes[1].set_ylim(-0.1, 1.1)
    
    fig.suptitle(f"Solver: {run.solver_name}", fontsize=12)
    plt.tight_layout()
    
    return fig


def create_leaderboard_plot(
    runs: List[EvaluationRun],
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Create a leaderboard-style plot.
    """
    # Sort by accuracy
    sorted_runs = sorted(runs, key=lambda r: r.accuracy, reverse=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    names = [r.solver_name for r in sorted_runs]
    scores = [r.accuracy * 100 for r in sorted_runs]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(scores)))
    
    bars = ax.barh(names, scores, color=colors)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f'{score:.1f}%', va='center', fontsize=10)
    
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Solver Leaderboard')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()  # Best at top
    
    plt.tight_layout()
    return fig
