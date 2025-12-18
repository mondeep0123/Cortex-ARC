"""Tests for solvers."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.grid import Grid
from core.task import Task, TaskPair
from solvers.brute_force import BruteForceSolver
from solvers.program_synthesis import ProgramSynthesisSolver


def create_rotation_task():
    """Create a simple rotation task for testing."""
    # Task: rotate 90 degrees
    train_pairs = [
        TaskPair(
            input=Grid.from_list([[1, 2], [3, 4]]),
            output=Grid.from_list([[3, 1], [4, 2]])
        ),
        TaskPair(
            input=Grid.from_list([[5, 6], [7, 8]]),
            output=Grid.from_list([[7, 5], [8, 6]])
        ),
    ]
    
    test_pairs = [
        TaskPair(
            input=Grid.from_list([[1, 0], [0, 1]]),
            output=Grid.from_list([[0, 1], [1, 0]])
        )
    ]
    
    return Task(
        task_id="test_rotation",
        train=train_pairs,
        test=test_pairs
    )


def create_flip_task():
    """Create a horizontal flip task."""
    train_pairs = [
        TaskPair(
            input=Grid.from_list([[1, 2, 3]]),
            output=Grid.from_list([[3, 2, 1]])
        ),
    ]
    
    test_pairs = [
        TaskPair(
            input=Grid.from_list([[4, 5, 6]]),
            output=Grid.from_list([[6, 5, 4]])
        )
    ]
    
    return Task(
        task_id="test_flip",
        train=train_pairs,
        test=test_pairs
    )


class TestBruteForceSolver:
    """Test brute force solver."""
    
    def test_solves_rotation(self):
        """Test that brute force can solve rotation tasks."""
        solver = BruteForceSolver()
        task = create_rotation_task()
        
        result = solver.solve(task)
        
        assert result.task_id == "test_rotation"
        assert len(result.predictions) == 1
        assert result.predictions[0][0] == task.test[0].output
    
    def test_solves_flip(self):
        """Test that brute force can solve flip tasks."""
        solver = BruteForceSolver()
        task = create_flip_task()
        
        result = solver.solve(task)
        
        assert len(result.predictions) == 1
        assert result.predictions[0][0] == task.test[0].output
    
    def test_returns_prediction_for_unsolvable(self):
        """Test that solver returns something even for unsolvable tasks."""
        # Create a task that can't be solved by simple transforms
        task = Task(
            task_id="unsolvable",
            train=[
                TaskPair(
                    input=Grid.from_list([[1, 2], [3, 4]]),
                    output=Grid.from_list([[9, 9, 9], [9, 9, 9]])  # Not a simple transform
                )
            ],
            test=[
                TaskPair(
                    input=Grid.from_list([[5, 6], [7, 8]]),
                    output=Grid.from_list([[9, 9, 9], [9, 9, 9]])
                )
            ]
        )
        
        solver = BruteForceSolver()
        result = solver.solve(task)
        
        # Should still return a prediction
        assert len(result.predictions) == 1
        assert len(result.predictions[0]) > 0


class TestProgramSynthesisSolver:
    """Test program synthesis solver."""
    
    def test_solves_simple_task(self):
        """Test that program synthesis can solve simple tasks."""
        solver = ProgramSynthesisSolver(max_depth=3, timeout=10.0)
        task = create_flip_task()
        
        result = solver.solve(task)
        
        assert len(result.predictions) == 1
    
    def test_returns_metadata(self):
        """Test that solver returns program information."""
        solver = ProgramSynthesisSolver(max_depth=2, timeout=5.0)
        task = create_rotation_task()
        
        result = solver.solve(task)
        
        assert "synthesized" in result.metadata


class TestSolverResult:
    """Test SolverResult functionality."""
    
    def test_to_submission_format(self):
        """Test conversion to Kaggle submission format."""
        solver = BruteForceSolver()
        task = create_rotation_task()
        
        result = solver.solve(task)
        submission = result.to_submission_format()
        
        assert "test_rotation" in submission
        assert len(submission["test_rotation"]) == 1  # 1 test case
        assert len(submission["test_rotation"][0]) == 2  # 2 attempts
    
    def test_get_best_predictions(self):
        """Test getting best predictions."""
        solver = BruteForceSolver()
        task = create_rotation_task()
        
        result = solver.solve(task)
        best = result.get_best_predictions()
        
        assert len(best) == 1
        assert isinstance(best[0], Grid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
