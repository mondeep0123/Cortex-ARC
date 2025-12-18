"""Tests for evaluation metrics."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.grid import Grid
from core.task import Task, TaskPair
from solvers.base import SolverResult
from evaluation.metrics import (
    exact_match,
    partial_match,
    shape_match,
    evaluate_task,
    accuracy
)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_exact_match_true(self):
        """Test exact match when grids are equal."""
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 2], [3, 4]])
        
        assert exact_match(g1, g2) is True
    
    def test_exact_match_false(self):
        """Test exact match when grids differ."""
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 2], [3, 5]])
        
        assert exact_match(g1, g2) is False
    
    def test_partial_match(self):
        """Test partial match calculation."""
        g1 = Grid.from_list([[1, 1, 1, 1]])
        g2 = Grid.from_list([[1, 1, 0, 0]])
        
        assert partial_match(g1, g2) == 0.5
    
    def test_partial_match_different_shapes(self):
        """Test partial match with different shapes."""
        g1 = Grid.from_list([[1, 2]])
        g2 = Grid.from_list([[1, 2, 3]])
        
        assert partial_match(g1, g2) == 0.0
    
    def test_shape_match(self):
        """Test shape matching."""
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[5, 6], [7, 8]])
        g3 = Grid.from_list([[1, 2, 3]])
        
        assert shape_match(g1, g2) is True
        assert shape_match(g1, g3) is False


class TestEvaluateTask:
    """Test task-level evaluation."""
    
    def test_evaluate_correct_predictions(self):
        """Test evaluation when predictions are correct."""
        task = Task(
            task_id="test",
            train=[],
            test=[
                TaskPair(
                    input=Grid.from_list([[1]]),
                    output=Grid.from_list([[2]])
                )
            ]
        )
        
        result = SolverResult(
            task_id="test",
            predictions=[[Grid.from_list([[2]])]],
            confidence=[[1.0]]
        )
        
        eval_result = evaluate_task(result, task)
        
        assert eval_result["accuracy"] == 1.0
        assert eval_result["correct"] == 1
    
    def test_evaluate_wrong_predictions(self):
        """Test evaluation when predictions are wrong."""
        task = Task(
            task_id="test",
            train=[],
            test=[
                TaskPair(
                    input=Grid.from_list([[1]]),
                    output=Grid.from_list([[2]])
                )
            ]
        )
        
        result = SolverResult(
            task_id="test",
            predictions=[[Grid.from_list([[3]])]],  # Wrong!
            confidence=[[0.5]]
        )
        
        eval_result = evaluate_task(result, task)
        
        assert eval_result["accuracy"] == 0.0
        assert eval_result["correct"] == 0
    
    def test_evaluate_second_attempt(self):
        """Test that second attempt counts if first is wrong."""
        task = Task(
            task_id="test",
            train=[],
            test=[
                TaskPair(
                    input=Grid.from_list([[1]]),
                    output=Grid.from_list([[2]])
                )
            ]
        )
        
        result = SolverResult(
            task_id="test",
            predictions=[[
                Grid.from_list([[3]]),  # Wrong
                Grid.from_list([[2]])   # Correct!
            ]],
            confidence=[[0.5, 0.3]]
        )
        
        eval_result = evaluate_task(result, task)
        
        # Should be correct because second attempt matches
        assert eval_result["accuracy"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
