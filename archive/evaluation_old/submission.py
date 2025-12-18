"""Submission utilities for Kaggle ARC Prize."""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from ..core.grid import Grid
from ..core.task import Task
from ..solvers.base import SolverResult


def generate_submission(
    results: List[SolverResult],
    output_path: Path = Path("submission.json")
) -> Path:
    """
    Generate Kaggle submission file.
    
    Format: Dict mapping task_id to list of [attempt1, attempt2] predictions.
    Each attempt is a 2D list of integers.
    
    Args:
        results: List of solver results
        output_path: Where to save submission
    
    Returns:
        Path to submission file
    """
    submission = {}
    
    for result in results:
        task_preds = []
        
        for test_preds in result.predictions:
            # Take up to 2 predictions per test case
            attempts = []
            
            for pred in test_preds[:2]:
                attempts.append(pred.to_list())
            
            # Pad with first prediction if needed
            while len(attempts) < 2:
                if attempts:
                    attempts.append(attempts[0])
                else:
                    attempts.append([[0]])  # Fallback
            
            task_preds.append(attempts)
        
        submission[result.task_id] = task_preds
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f)
    
    print(f"Submission saved to {output_path}")
    print(f"Total tasks: {len(submission)}")
    
    return output_path


def validate_submission(
    submission_path: Path,
    tasks: List[Task]
) -> Dict[str, Any]:
    """
    Validate a submission file.
    
    Checks:
    - All required task IDs present
    - Correct number of test predictions
    - Valid grid format (2D list of ints 0-9)
    - Grid dimensions within limits (1-30)
    
    Returns dict with validation results.
    """
    with open(submission_path, "r") as f:
        submission = json.load(f)
    
    errors = []
    warnings = []
    
    task_ids = {t.task_id for t in tasks}
    submission_ids = set(submission.keys())
    
    # Check missing/extra tasks
    missing = task_ids - submission_ids
    extra = submission_ids - task_ids
    
    if missing:
        errors.append(f"Missing tasks: {missing}")
    if extra:
        warnings.append(f"Extra tasks (will be ignored): {extra}")
    
    # Validate each task
    for task_id, predictions in submission.items():
        if task_id not in task_ids:
            continue
        
        task = next(t for t in tasks if t.task_id == task_id)
        expected_tests = len(task.test)
        
        if len(predictions) != expected_tests:
            errors.append(
                f"{task_id}: Expected {expected_tests} test predictions, got {len(predictions)}"
            )
            continue
        
        for test_idx, attempts in enumerate(predictions):
            if len(attempts) != 2:
                errors.append(
                    f"{task_id} test {test_idx}: Expected 2 attempts, got {len(attempts)}"
                )
                continue
            
            for attempt_idx, grid in enumerate(attempts):
                # Validate grid
                if not isinstance(grid, list):
                    errors.append(
                        f"{task_id} test {test_idx} attempt {attempt_idx}: Not a list"
                    )
                    continue
                
                if not grid:
                    errors.append(
                        f"{task_id} test {test_idx} attempt {attempt_idx}: Empty grid"
                    )
                    continue
                
                height = len(grid)
                if height > 30:
                    errors.append(
                        f"{task_id} test {test_idx} attempt {attempt_idx}: Height {height} > 30"
                    )
                
                for row_idx, row in enumerate(grid):
                    if not isinstance(row, list):
                        errors.append(
                            f"{task_id} test {test_idx} attempt {attempt_idx}: Row {row_idx} not a list"
                        )
                        continue
                    
                    width = len(row)
                    if width > 30:
                        errors.append(
                            f"{task_id} test {test_idx} attempt {attempt_idx}: Width {width} > 30"
                        )
                    
                    for col_idx, val in enumerate(row):
                        if not isinstance(val, int) or val < 0 or val > 9:
                            errors.append(
                                f"{task_id} test {test_idx} attempt {attempt_idx}: "
                                f"Invalid value {val} at ({row_idx}, {col_idx})"
                            )
    
    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_tasks": len(submission),
        "expected_tasks": len(tasks),
    }
    
    # Print summary
    if errors:
        print(f"❌ Submission INVALID - {len(errors)} errors:")
        for e in errors[:10]:  # Show first 10
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print(f"✅ Submission valid - {len(submission)} tasks")
    
    if warnings:
        print(f"⚠️ {len(warnings)} warnings:")
        for w in warnings:
            print(f"  - {w}")
    
    return result


def load_submission(path: Path) -> Dict[str, List[List[Grid]]]:
    """
    Load a submission file and convert to Grids.
    
    Returns dict mapping task_id to list of [Grid, Grid] attempts per test.
    """
    with open(path, "r") as f:
        submission = json.load(f)
    
    result = {}
    for task_id, predictions in submission.items():
        task_preds = []
        for attempts in predictions:
            attempt_grids = [Grid.from_list(g) for g in attempts]
            task_preds.append(attempt_grids)
        result[task_id] = task_preds
    
    return result
