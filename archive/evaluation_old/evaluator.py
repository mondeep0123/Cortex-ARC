"""Evaluator class for ARC-AGI benchmarking."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from tqdm import tqdm

from ..core.task import Task
from ..data.loader import ARCDataset
from ..solvers.base import Solver, SolverResult
from .metrics import evaluate_task, accuracy


@dataclass
class EvaluationRun:
    """Results of an evaluation run."""
    
    solver_name: str
    dataset_name: str
    timestamp: str
    
    # Overall metrics
    accuracy: float
    partial_accuracy: float
    shape_accuracy: float
    
    # Per-task results
    task_results: Dict[str, Dict[str, Any]]
    
    # Timing
    total_time: float
    avg_time_per_task: float
    
    # Metadata
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_name": self.solver_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "accuracy": self.accuracy,
            "partial_accuracy": self.partial_accuracy,
            "shape_accuracy": self.shape_accuracy,
            "total_time": self.total_time,
            "avg_time_per_task": self.avg_time_per_task,
            "task_results": self.task_results,
            "config": self.config,
        }
    
    def save(self, path: Path):
        """Save results to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> EvaluationRun:
        """Load results from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"=== Evaluation Results ===",
            f"Solver: {self.solver_name}",
            f"Dataset: {self.dataset_name}",
            f"Timestamp: {self.timestamp}",
            f"",
            f"Accuracy: {self.accuracy:.2%}",
            f"Partial Accuracy: {self.partial_accuracy:.2%}",
            f"Shape Accuracy: {self.shape_accuracy:.2%}",
            f"",
            f"Total Time: {self.total_time:.1f}s",
            f"Avg Time/Task: {self.avg_time_per_task:.2f}s",
            f"",
            f"Tasks Solved: {sum(1 for r in self.task_results.values() if r['accuracy'] == 1.0)}/"
            f"{len(self.task_results)}",
        ]
        return "\n".join(lines)


class Evaluator:
    """
    Main evaluator class for benchmarking solvers on ARC-AGI.
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        verbose: bool = True
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("experiments/logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
    
    def evaluate(
        self,
        solver: Solver,
        dataset: ARCDataset,
        timeout_per_task: float = 60.0,
        save_results: bool = True
    ) -> EvaluationRun:
        """
        Evaluate a solver on a dataset.
        
        Args:
            solver: The solver to evaluate
            dataset: The dataset to evaluate on
            timeout_per_task: Maximum time per task (seconds)
            save_results: Whether to save results to disk
        
        Returns:
            EvaluationRun with all results
        """
        from datetime import datetime
        
        timestamp = datetime.now().isoformat()
        task_results = {}
        all_results = []
        
        start_time = time.time()
        
        # Evaluate each task
        tasks = list(dataset)
        iterator = tqdm(tasks, desc=f"Evaluating {solver.name}") if self.verbose else tasks
        
        for task in iterator:
            try:
                # Solve with timeout
                result = solver.solve(task)
                
                # Evaluate
                eval_result = evaluate_task(result, task)
                task_results[task.task_id] = eval_result
                all_results.append(result)
                
                if self.verbose:
                    iterator.set_postfix({
                        "acc": f"{eval_result['accuracy']:.0%}",
                        "time": f"{eval_result['solve_time']:.1f}s"
                    })
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error on {task.task_id}: {e}")
                task_results[task.task_id] = {
                    "task_id": task.task_id,
                    "error": str(e),
                    "accuracy": 0.0,
                    "partial_accuracy": 0.0,
                    "shape_accuracy": 0.0,
                }
        
        total_time = time.time() - start_time
        
        # Aggregate metrics
        valid_results = [r for r in task_results.values() if "error" not in r]
        
        avg_accuracy = sum(r["accuracy"] for r in valid_results) / len(valid_results) if valid_results else 0
        avg_partial = sum(r["partial_accuracy"] for r in valid_results) / len(valid_results) if valid_results else 0
        avg_shape = sum(r["shape_accuracy"] for r in valid_results) / len(valid_results) if valid_results else 0
        
        # Create run object
        run = EvaluationRun(
            solver_name=solver.name,
            dataset_name=f"{dataset.version}/{dataset.split}",
            timestamp=timestamp,
            accuracy=avg_accuracy,
            partial_accuracy=avg_partial,
            shape_accuracy=avg_shape,
            task_results=task_results,
            total_time=total_time,
            avg_time_per_task=total_time / len(tasks) if tasks else 0,
            config=solver.config
        )
        
        # Save if requested
        if save_results:
            filename = f"{solver.name}_{dataset.version}_{dataset.split}_{timestamp.replace(':', '-')}.json"
            run.save(self.output_dir / filename)
        
        if self.verbose:
            print(run.summary())
        
        return run
    
    def compare(
        self,
        solvers: List[Solver],
        dataset: ARCDataset
    ) -> Dict[str, EvaluationRun]:
        """
        Compare multiple solvers on the same dataset.
        
        Returns dict mapping solver name to results.
        """
        results = {}
        
        for solver in solvers:
            run = self.evaluate(solver, dataset)
            results[solver.name] = run
        
        # Print comparison
        if self.verbose:
            print("\n=== Comparison ===")
            print(f"{'Solver':<20} {'Accuracy':>10} {'Partial':>10} {'Time':>10}")
            print("-" * 52)
            for name, run in results.items():
                print(f"{name:<20} {run.accuracy:>9.2%} {run.partial_accuracy:>9.2%} {run.total_time:>9.1f}s")
        
        return results
