"""Evaluate a solver on ARC-AGI datasets."""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_arc1, load_arc2, load_from_directory
from data.loader import ARCDataset
from solvers.brute_force import BruteForceSolver
from solvers.program_synthesis import ProgramSynthesisSolver
from evaluation.evaluator import Evaluator


def get_solver(name: str, **kwargs):
    """Get solver by name."""
    solvers = {
        "brute_force": BruteForceSolver,
        "program_synthesis": ProgramSynthesisSolver,
    }
    
    if name not in solvers:
        raise ValueError(f"Unknown solver: {name}. Available: {list(solvers.keys())}")
    
    return solvers[name](**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC-AGI solver")
    
    parser.add_argument(
        "--solver",
        type=str,
        default="brute_force",
        help="Solver to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample",
        choices=["arc1", "arc2", "sample"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "evaluation"],
        help="Dataset split"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/logs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout per task in seconds"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    data_dir = Path(args.data_dir)
    
    if args.dataset == "arc1":
        datasets = load_arc1(data_dir / "arc-agi-1", splits=[args.split])
        dataset = datasets.get(args.split)
    elif args.dataset == "arc2":
        datasets = load_arc2(data_dir / "arc-agi-2", splits=[args.split])
        dataset = datasets.get(args.split)
    else:  # sample
        sample_dir = data_dir / "sample" / args.split
        if not sample_dir.exists():
            print(f"Sample dataset not found. Run: python scripts/download_data.py --version sample")
            return
        dataset = load_from_directory(sample_dir, version="sample", split=args.split)
    
    if dataset is None:
        print(f"Dataset not found at {data_dir}")
        return
    
    print(f"Loaded {len(dataset)} tasks")
    
    # Limit tasks if requested
    if args.max_tasks and args.max_tasks < len(dataset):
        task_ids = list(dataset.tasks.keys())[:args.max_tasks]
        dataset = ARCDataset(
            tasks={tid: dataset.tasks[tid] for tid in task_ids},
            version=dataset.version,
            split=dataset.split
        )
        print(f"Limited to {len(dataset)} tasks")
    
    # Create solver
    solver = get_solver(args.solver)
    print(f"Using solver: {solver.name}")
    
    # Evaluate
    evaluator = Evaluator(output_dir=Path(args.output_dir))
    run = evaluator.evaluate(
        solver=solver,
        dataset=dataset,
        timeout_per_task=args.timeout
    )
    
    print("\n" + run.summary())


if __name__ == "__main__":
    main()
