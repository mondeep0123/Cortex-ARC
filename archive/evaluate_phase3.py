"""
Phase 3 Evaluation Script

Tests the Phase3Solver on:
1. Synthetic test cases (built-in)
2. Real ARC puzzles (if data is available)

Usage:
    python scripts/evaluate_phase3.py              # Run synthetic tests only
    python scripts/evaluate_phase3.py --arc        # Run on real ARC data
    python scripts/evaluate_phase3.py --arc --n 50 # Run on 50 random puzzles
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain.visual.solver import Phase3Solver, Task, TrainExample, create_test_task


# ============================================================================
# SYNTHETIC TEST CASES
# ============================================================================

SYNTHETIC_TESTS = [
    # Test 1: Translation (move right by 1)
    {
        "name": "translate_right",
        "train": [
            ([[0, 1, 0, 0], [0, 0, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 0]]),
            ([[1, 0, 0, 0], [0, 0, 0, 0]], [[0, 1, 0, 0], [0, 0, 0, 0]]),
        ],
        "test_in": [[0, 0, 1, 0], [0, 0, 0, 0]],
        "test_out": [[0, 0, 0, 1], [0, 0, 0, 0]],
        "type": "translation"
    },
    # Test 2: Translation (move down by 1)
    {
        "name": "translate_down",
        "train": [
            ([[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        ],
        "test_in": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        "test_out": [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        "type": "translation"
    },
    # Test 3: Recolor (blue to red)
    {
        "name": "recolor_1_to_2",
        "train": [
            ([[0, 1, 0], [0, 1, 0]], [[0, 2, 0], [0, 2, 0]]),
        ],
        "test_in": [[1, 1, 0], [0, 0, 0]],
        "test_out": [[2, 2, 0], [0, 0, 0]],
        "type": "recolor"
    },
    # Test 4: Identity (no change)
    {
        "name": "identity",
        "train": [
            ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ],
        "test_in": [[5, 6], [7, 8]],
        "test_out": [[5, 6], [7, 8]],
        "type": "identity"
    },
    # Test 5: Horizontal flip
    {
        "name": "flip_horizontal",
        "train": [
            ([[1, 2, 3], [4, 5, 6]], [[3, 2, 1], [6, 5, 4]]),
        ],
        "test_in": [[0, 1, 0], [2, 0, 3]],
        "test_out": [[0, 1, 0], [3, 0, 2]],
        "type": "flip"
    },
    # Test 6: Vertical flip
    {
        "name": "flip_vertical",
        "train": [
            ([[1, 2], [3, 4], [5, 6]], [[5, 6], [3, 4], [1, 2]]),
        ],
        "test_in": [[0, 1], [2, 0], [3, 0]],
        "test_out": [[3, 0], [2, 0], [0, 1]],
        "type": "flip"
    },
    # Test 7: Rotate 90° clockwise
    {
        "name": "rotate_90",
        "train": [
            ([[1, 2], [3, 4], [5, 6]], [[5, 3, 1], [6, 4, 2]]),
        ],
        "test_in": [[0, 1], [2, 0], [0, 3]],
        "test_out": [[0, 2, 0], [3, 0, 1]],
        "type": "rotation"
    },
    # Test 8: Rotate 180°
    {
        "name": "rotate_180",
        "train": [
            ([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]),
        ],
        "test_in": [[0, 1, 0], [2, 0, 3]],
        "test_out": [[3, 0, 2], [0, 1, 0]],
        "type": "rotation"
    },
    # Test 9: Multiple objects, same translation (all move right by 1)
    {
        "name": "translate_multiple_objects",
        "train": [
            ([[1, 0, 2, 0], [0, 0, 0, 0]], [[0, 1, 0, 2], [0, 0, 0, 0]]),  # Both move right by 1
        ],
        "test_in": [[1, 0, 0, 2], [0, 0, 0, 0]],  # Blue at 0, red at 3
        "test_out": [[0, 1, 0, 0], [0, 0, 0, 0]],  # Blue to 1, red clips (goes to col 4, out of bounds)
        "type": "translation"
    },
    # Test 10: Scale 2x
    {
        "name": "scale_2x",
        "train": [
            ([[1, 2], [3, 0]], [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 0, 0], [3, 3, 0, 0]]),
        ],
        "test_in": [[1, 0], [0, 2]],
        "test_out": [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]],
        "type": "scale"
    },
    # Test 11: Tile 2x2
    {
        "name": "tile_2x",
        "train": [
            ([[1, 2], [3, 4]], [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]),
        ],
        "test_in": [[5, 6], [7, 8]],
        "test_out": [[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]],
        "type": "tile"
    },
    # Test 12: Transpose
    {
        "name": "transpose",
        "train": [
            ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
        ],
        "test_in": [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
        "test_out": [[1, 0, 0], [0, 2, 0], [0, 0, 3]],  # Diagonal is same when transposed
        "type": "transpose"
    },
    # Test 13: Crop
    {
        "name": "crop",
        "train": [
            ([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]], [[1, 2], [3, 4]]),
        ],
        "test_in": [[0, 0, 0], [0, 5, 0], [0, 0, 0]],
        "test_out": [[5]],
        "type": "crop"
    },
]


def run_synthetic_tests(solver: Phase3Solver) -> Tuple[int, int, List[dict]]:
    """Run all synthetic tests and return results."""
    passed = 0
    failed = 0
    results = []
    
    print("\n" + "=" * 60)
    print("SYNTHETIC TEST SUITE")
    print("=" * 60)
    
    for test in SYNTHETIC_TESTS:
        task = create_test_task(
            test["name"],
            test["train"],
            test["test_in"],
            test["test_out"]
        )
        
        is_correct, accuracy = solver.evaluate(task)
        
        status = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"  {status}: {test['name']} ({test['type']}) - {accuracy:.1%}")
        
        if is_correct:
            passed += 1
        else:
            failed += 1
            # Show expected vs actual for failed tests
            prediction = solver.solve(task)
            print(f"       Expected: {task.test_output.tolist()}")
            print(f"       Got:      {prediction.tolist()}")
        
        results.append({
            "name": test["name"],
            "type": test["type"],
            "correct": is_correct,
            "accuracy": accuracy
        })
    
    print("-" * 60)
    print(f"TOTAL: {passed}/{passed + failed} passed ({100*passed/(passed+failed):.1f}%)")
    print("=" * 60)
    
    return passed, failed, results


# ============================================================================
# REAL ARC EVALUATION
# ============================================================================

def load_arc_task(json_path: Path) -> Optional[Task]:
    """Load a single ARC task from JSON file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        train = []
        for ex in data.get("train", []):
            train.append(TrainExample(
                input_grid=np.array(ex["input"], dtype=np.int8),
                output_grid=np.array(ex["output"], dtype=np.int8)
            ))
        
        test_cases = data.get("test", [])
        if not test_cases:
            return None
        
        # Use first test case
        test_input = np.array(test_cases[0]["input"], dtype=np.int8)
        test_output = np.array(test_cases[0]["output"], dtype=np.int8) if "output" in test_cases[0] else None
        
        return Task(
            task_id=json_path.stem,
            train=train,
            test_input=test_input,
            test_output=test_output
        )
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def run_arc_evaluation(solver: Phase3Solver, data_dir: Path, n_puzzles: int = 50) -> Tuple[int, int, List[dict]]:
    """Evaluate on real ARC puzzles."""
    
    # Find all JSON files
    json_files = list(data_dir.glob("**/*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return 0, 0, []
    
    import random
    random.shuffle(json_files)
    json_files = json_files[:n_puzzles]
    
    print("\n" + "=" * 60)
    print(f"ARC EVALUATION ({len(json_files)} puzzles)")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    results = []
    
    for json_path in json_files:
        task = load_arc_task(json_path)
        
        if task is None or task.test_output is None:
            skipped += 1
            continue
        
        try:
            is_correct, accuracy = solver.evaluate(task)
            
            if is_correct:
                passed += 1
                print(f"  ✓ {task.task_id}")
            else:
                failed += 1
                if accuracy > 0.5:
                    print(f"  ~ {task.task_id} ({accuracy:.1%} correct)")
                else:
                    print(f"  ✗ {task.task_id}")
            
            results.append({
                "task_id": task.task_id,
                "correct": is_correct,
                "accuracy": accuracy
            })
        except Exception as e:
            print(f"  ! {task.task_id}: {str(e)[:50]}")
            skipped += 1
    
    total = passed + failed
    print("-" * 60)
    print(f"RESULTS: {passed}/{total} correct ({100*passed/total if total > 0 else 0:.1f}%)")
    print(f"         {skipped} skipped (no test output or error)")
    print("=" * 60)
    
    return passed, failed, results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Solver Evaluation")
    parser.add_argument("--arc", action="store_true", help="Run on real ARC puzzles")
    parser.add_argument("--n", type=int, default=50, help="Number of ARC puzzles to test")
    parser.add_argument("--data", type=str, default="data/arc-agi-1/training", help="Path to ARC data")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PHASE 3 SOLVER EVALUATION")
    print("=" * 60)
    
    solver = Phase3Solver()
    
    # Always run synthetic tests
    syn_passed, syn_failed, syn_results = run_synthetic_tests(solver)
    
    # Optionally run ARC evaluation
    if args.arc:
        data_dir = Path(args.data)
        if data_dir.exists():
            arc_passed, arc_failed, arc_results = run_arc_evaluation(solver, data_dir, args.n)
        else:
            print(f"\nARC data not found at {data_dir}")
            print("To download ARC data, run:")
            print("  python scripts/download_data.py")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
