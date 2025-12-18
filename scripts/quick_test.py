"""Complete test of all transforms - verbose."""
import sys
sys.path.insert(0, '.')

from src.brain.visual.solver import Phase3Solver, create_test_task

solver = Phase3Solver()

tests = [
    # Original tests
    ("translate_right", [
        ([[0, 1, 0, 0], [0, 0, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 0]]),
        ([[1, 0, 0, 0], [0, 0, 0, 0]], [[0, 1, 0, 0], [0, 0, 0, 0]]),
    ], [[0, 0, 1, 0], [0, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 0, 0]]),
    
    ("rotate_90", [
        ([[1, 2], [3, 4], [5, 6]], [[5, 3, 1], [6, 4, 2]])
    ], [[0, 1], [2, 0], [0, 3]], [[0, 2, 0], [3, 0, 1]]),
    
    ("rotate_180", [
        ([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]])
    ], [[0, 1, 0], [2, 0, 3]], [[3, 0, 2], [0, 1, 0]]),
    
    ("flip_h", [
        ([[1, 2, 3], [4, 5, 6]], [[3, 2, 1], [6, 5, 4]])
    ], [[0, 1, 0], [2, 0, 3]], [[0, 1, 0], [3, 0, 2]]),
    
    ("flip_v", [
        ([[1, 2], [3, 4], [5, 6]], [[5, 6], [3, 4], [1, 2]])
    ], [[0, 1], [2, 0], [3, 0]], [[3, 0], [2, 0], [0, 1]]),
    
    ("identity", [
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    ], [[5, 6], [7, 8]], [[5, 6], [7, 8]]),
    
    ("recolor", [
        ([[0, 1, 0], [0, 1, 0]], [[0, 2, 0], [0, 2, 0]])
    ], [[1, 1, 0], [0, 0, 0]], [[2, 2, 0], [0, 0, 0]]),
    
    # New transforms
    ("scale_2x", [
        ([[1, 2], [3, 0]], [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 0, 0], [3, 3, 0, 0]])
    ], [[1, 0], [0, 2]], [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]]),
    
    ("tile_2x", [
        ([[1, 2], [3, 4]], [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])
    ], [[5, 6], [7, 8]], [[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]]),
    
    ("crop", [
        ([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]], [[1, 2], [3, 4]])
    ], [[0, 0, 0], [0, 5, 0], [0, 0, 0]], [[5]]),
    
    ("transpose", [
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    ], [[1, 0, 0], [0, 2, 0], [0, 0, 3]], [[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
]

passed = 0
failed = 0

print("=" * 60)
print("CORTEX-ARC TRANSFORM TESTS")
print("=" * 60)

for name, train, test_in, test_out in tests:
    task = create_test_task(name, train, test_in, test_out)
    is_correct, accuracy = solver.evaluate(task)
    status = "PASS" if is_correct else "FAIL"
    print(f"  {status}: {name} - {accuracy:.1%}")
    if is_correct:
        passed += 1
    else:
        failed += 1
        pred = solver.solve(task)
        print(f"    Expected: {task.test_output.tolist()}")
        print(f"    Got:      {pred.tolist()}")

print("-" * 60)
print(f"TOTAL: {passed}/{passed+failed} passed ({100*passed/(passed+failed):.1f}%)")
print("=" * 60)
