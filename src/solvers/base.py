"""Base solver interface for ARC-AGI."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import time

from ..core.grid import Grid
from ..core.task import Task


@dataclass
class SolverResult:
    """
    Result of solving a task.
    
    For ARC-AGI, we're allowed 2 attempts per test case.
    """
    
    task_id: str
    predictions: List[List[Grid]]  # For each test case, up to 2 predictions
    confidence: List[List[float]]  # Confidence scores for each prediction
    solve_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_test_cases(self) -> int:
        return len(self.predictions)
    
    def get_best_predictions(self) -> List[Grid]:
        """Get the highest-confidence prediction for each test case."""
        best = []
        for preds, confs in zip(self.predictions, self.confidence):
            if preds:
                best_idx = max(range(len(confs)), key=lambda i: confs[i])
                best.append(preds[best_idx])
            else:
                # Return empty grid as fallback
                best.append(Grid.zeros(1, 1))
        return best
    
    def to_submission_format(self) -> Dict[str, List[List[List[int]]]]:
        """
        Convert to Kaggle submission format.
        
        Returns dict mapping task_id to list of predictions (as nested lists).
        """
        preds = []
        for test_preds in self.predictions:
            # Take up to 2 predictions per test case
            test_preds_list = [
                p.to_list() for p in test_preds[:2]
            ]
            # Pad with first prediction if we only have 1
            while len(test_preds_list) < 2:
                if test_preds_list:
                    test_preds_list.append(test_preds_list[0])
                else:
                    test_preds_list.append([[0]])
            preds.append(test_preds_list)
        
        return {self.task_id: preds}


class Solver(ABC):
    """
    Abstract base class for ARC-AGI solvers.
    
    A solver takes a task and produces predictions for the test cases.
    """
    
    def __init__(self, name: str = "base_solver", **kwargs):
        self.name = name
        self.config = kwargs
    
    @abstractmethod
    def solve(self, task: Task) -> SolverResult:
        """
        Solve a task.
        
        Args:
            task: The ARC task to solve
        
        Returns:
            SolverResult containing predictions
        """
        pass
    
    def solve_batch(self, tasks: List[Task]) -> List[SolverResult]:
        """
        Solve multiple tasks.
        
        Default implementation just calls solve() for each task.
        Override for parallel processing.
        """
        return [self.solve(task) for task in tasks]
    
    def _time_solve(self, task: Task) -> Tuple[SolverResult, float]:
        """Helper to time the solve operation."""
        start = time.time()
        result = self.solve(task)
        elapsed = time.time() - start
        result.solve_time = elapsed
        return result, elapsed
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class EnsembleSolver(Solver):
    """
    Ensemble of multiple solvers.
    
    Combines predictions from multiple solvers using voting or confidence.
    """
    
    def __init__(
        self,
        solvers: List[Solver],
        strategy: str = "voting",
        name: str = "ensemble"
    ):
        super().__init__(name=name)
        self.solvers = solvers
        self.strategy = strategy  # "voting", "confidence", "first_success"
    
    def solve(self, task: Task) -> SolverResult:
        """Solve using ensemble of solvers."""
        all_results = [solver.solve(task) for solver in self.solvers]
        
        # Combine predictions for each test case
        combined_predictions = []
        combined_confidence = []
        
        for test_idx in range(len(task.test)):
            # Collect all predictions for this test case
            test_predictions = []
            test_confidences = []
            
            for result in all_results:
                if test_idx < len(result.predictions):
                    test_predictions.extend(result.predictions[test_idx])
                    test_confidences.extend(result.confidence[test_idx])
            
            if self.strategy == "first_success":
                # Just take the first prediction
                combined_predictions.append(test_predictions[:2])
                combined_confidence.append(test_confidences[:2])
            
            elif self.strategy == "voting":
                # Vote among predictions (pick most common)
                from collections import Counter
                grid_hashes = [g.to_tuple() for g in test_predictions]
                counter = Counter(grid_hashes)
                
                # Get top 2 most common
                top_2 = counter.most_common(2)
                top_grids = []
                top_conf = []
                
                for grid_tuple, count in top_2:
                    # Reconstruct grid from tuple
                    import numpy as np
                    h = len(grid_tuple)
                    w = len(grid_tuple[0]) if h > 0 else 0
                    data = np.array(grid_tuple, dtype=np.int8)
                    top_grids.append(Grid(data=data))
                    top_conf.append(count / len(test_predictions))
                
                combined_predictions.append(top_grids)
                combined_confidence.append(top_conf)
            
            elif self.strategy == "confidence":
                # Sort by confidence and take top 2
                sorted_pairs = sorted(
                    zip(test_predictions, test_confidences),
                    key=lambda x: x[1],
                    reverse=True
                )
                top_2 = sorted_pairs[:2]
                combined_predictions.append([p[0] for p in top_2])
                combined_confidence.append([p[1] for p in top_2])
            
            else:
                combined_predictions.append(test_predictions[:2])
                combined_confidence.append(test_confidences[:2])
        
        return SolverResult(
            task_id=task.task_id,
            predictions=combined_predictions,
            confidence=combined_confidence,
            metadata={"ensemble_size": len(self.solvers), "strategy": self.strategy}
        )


class RefinementSolver(Solver):
    """
    Solver that iteratively refines predictions.
    
    This implements the "refinement loop" approach that won ARC Prize 2025.
    """
    
    def __init__(
        self,
        base_solver: Solver,
        refiner,  # Callable that improves predictions
        max_iterations: int = 5,
        name: str = "refinement"
    ):
        super().__init__(name=name)
        self.base_solver = base_solver
        self.refiner = refiner
        self.max_iterations = max_iterations
    
    def solve(self, task: Task) -> SolverResult:
        """Solve with iterative refinement."""
        # Get initial predictions
        result = self.base_solver.solve(task)
        
        # Refine each prediction
        for iteration in range(self.max_iterations):
            improved = False
            
            for test_idx, (preds, ground_truth) in enumerate(
                zip(result.predictions, task.test_outputs)
            ):
                for pred_idx, pred in enumerate(preds):
                    # Try to refine this prediction
                    refined = self.refiner(
                        task=task,
                        prediction=pred,
                        test_input=task.test[test_idx].input,
                        iteration=iteration
                    )
                    
                    if refined is not None and refined != pred:
                        result.predictions[test_idx][pred_idx] = refined
                        improved = True
            
            if not improved:
                break
        
        result.metadata["refinement_iterations"] = iteration + 1
        return result
