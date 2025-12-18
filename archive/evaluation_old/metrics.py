"""Evaluation metrics for ARC-AGI."""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

from ..core.grid import Grid
from ..core.task import Task
from ..solvers.base import SolverResult


def exact_match(prediction: Grid, ground_truth: Grid) -> bool:
    """
    Check if prediction exactly matches ground truth.
    
    This is the official ARC-AGI metric.
    """
    return prediction == ground_truth


def partial_match(prediction: Grid, ground_truth: Grid) -> float:
    """
    Calculate partial match score (fraction of cells correct).
    
    Useful for debugging and understanding near-misses.
    """
    if prediction.shape != ground_truth.shape:
        return 0.0
    
    return np.mean(prediction.data == ground_truth.data)


def shape_match(prediction: Grid, ground_truth: Grid) -> bool:
    """Check if shape matches."""
    return prediction.shape == ground_truth.shape


def color_accuracy(prediction: Grid, ground_truth: Grid) -> float:
    """
    Calculate color accuracy (ignoring position).
    
    Useful for understanding if the model learned the right colors.
    """
    if prediction.shape != ground_truth.shape:
        return 0.0
    
    pred_colors = set(prediction.unique_colors())
    true_colors = set(ground_truth.unique_colors())
    
    if not true_colors:
        return 1.0 if not pred_colors else 0.0
    
    intersection = pred_colors & true_colors
    union = pred_colors | true_colors
    
    return len(intersection) / len(union)


def evaluate_task(
    result: SolverResult,
    task: Task,
    num_attempts: int = 2
) -> Dict[str, any]:
    """
    Evaluate solver result against task ground truth.
    
    Args:
        result: Solver predictions
        task: Task with ground truth
        num_attempts: Number of attempts allowed (ARC allows 2)
    
    Returns:
        Dictionary with evaluation metrics
    """
    scores = []
    partial_scores = []
    shape_matches = []
    
    for test_idx, (preds, ground_truth) in enumerate(
        zip(result.predictions, task.test_outputs)
    ):
        # Check up to num_attempts predictions
        test_preds = preds[:num_attempts]
        
        # Check for exact match in any attempt
        matched = any(exact_match(p, ground_truth) for p in test_preds)
        scores.append(1 if matched else 0)
        
        # Best partial score
        if test_preds:
            best_partial = max(partial_match(p, ground_truth) for p in test_preds)
            partial_scores.append(best_partial)
            
            # Shape match in any attempt
            shape_matched = any(shape_match(p, ground_truth) for p in test_preds)
            shape_matches.append(shape_matched)
        else:
            partial_scores.append(0.0)
            shape_matches.append(False)
    
    return {
        "task_id": task.task_id,
        "num_test": len(scores),
        "correct": sum(scores),
        "accuracy": sum(scores) / len(scores) if scores else 0.0,
        "partial_accuracy": sum(partial_scores) / len(partial_scores) if partial_scores else 0.0,
        "shape_accuracy": sum(shape_matches) / len(shape_matches) if shape_matches else 0.0,
        "solve_time": result.solve_time,
        "per_test": scores,
    }


def accuracy(results: List[SolverResult], tasks: List[Task]) -> float:
    """
    Calculate overall accuracy over multiple tasks.
    
    This is the primary ARC-AGI competition metric.
    """
    total_correct = 0
    total_tests = 0
    
    task_dict = {t.task_id: t for t in tasks}
    
    for result in results:
        if result.task_id in task_dict:
            eval_result = evaluate_task(result, task_dict[result.task_id])
            total_correct += eval_result["correct"]
            total_tests += eval_result["num_test"]
    
    return total_correct / total_tests if total_tests > 0 else 0.0


def compute_confusion_matrix(
    results: List[SolverResult],
    tasks: List[Task]
) -> Dict[str, int]:
    """
    Compute confusion statistics.
    
    Returns counts of:
    - correct: Exact matches
    - partial: Shape correct but content wrong
    - wrong_shape: Wrong output shape
    - no_prediction: No prediction made
    """
    stats = {
        "correct": 0,
        "partial": 0,
        "wrong_shape": 0,
        "no_prediction": 0,
        "total": 0
    }
    
    task_dict = {t.task_id: t for t in tasks}
    
    for result in results:
        if result.task_id not in task_dict:
            continue
        
        task = task_dict[result.task_id]
        
        for preds, ground_truth in zip(result.predictions, task.test_outputs):
            stats["total"] += 1
            
            if not preds:
                stats["no_prediction"] += 1
                continue
            
            best_pred = preds[0]
            
            if exact_match(best_pred, ground_truth):
                stats["correct"] += 1
            elif shape_match(best_pred, ground_truth):
                stats["partial"] += 1
            else:
                stats["wrong_shape"] += 1
    
    return stats
