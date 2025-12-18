"""Generate Kaggle submission file."""

import argparse
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_arc1, load_arc2, load_from_directory
from solvers.brute_force import BruteForceSolver
from solvers.program_synthesis import ProgramSynthesisSolver
from solvers.base import EnsembleSolver
from evaluation.submission import generate_submission, validate_submission


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission")
    
    parser.add_argument(
        "--solver",
        type=str,
        default="ensemble",
        help="Solver to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="arc2",
        help="Dataset (arc1, arc2)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.json",
        help="Output submission file"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate submission after generation"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Load test dataset
    if args.dataset == "arc1":
        datasets = load_arc1(data_dir / "arc-agi-1", splits=["evaluation"])
        dataset = datasets.get("evaluation")
    else:
        datasets = load_arc2(data_dir / "arc-agi-2", splits=["evaluation"])
        dataset = datasets.get("evaluation")
    
    if dataset is None:
        print(f"Dataset not found. Please download first.")
        return
    
    # Create solver
    if args.solver == "ensemble":
        solver = EnsembleSolver(
            solvers=[
                BruteForceSolver(),
                ProgramSynthesisSolver(),
            ],
            strategy="voting"
        )
    elif args.solver == "brute_force":
        solver = BruteForceSolver()
    elif args.solver == "program_synthesis":
        solver = ProgramSynthesisSolver()
    else:
        print(f"Unknown solver: {args.solver}")
        return
    
    print(f"Solving {len(dataset)} tasks with {solver.name}...")
    
    # Generate predictions
    from tqdm import tqdm
    
    results = []
    for task in tqdm(dataset, desc="Solving"):
        result = solver.solve(task)
        results.append(result)
    
    # Generate submission
    output_path = generate_submission(results, Path(args.output))
    
    # Validate if requested
    if args.validate:
        print("\nValidating submission...")
        tasks = list(dataset)
        validate_submission(output_path, tasks)


if __name__ == "__main__":
    main()
