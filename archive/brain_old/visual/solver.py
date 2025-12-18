"""
Phase 3 Solver - Minimal Viable Pipeline

The complete Phase 3 solver that:
1. Processes grids with VisualCortex
2. Matches objects with ObjectMatcher
3. Detects transformations with ComparisonModule
4. Extracts rules and applies to test input

~200 lines total. Minimal, efficient, testable.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

try:
    from .perception import VisualCortex, VisualOutput
    from .reasoning import (
        ObjectMatcher, 
        ComparisonModule, 
        TransformationSignature,
        TransformationRule,
        extract_rules,
        TransformType
    )
except ImportError:
    from perception import VisualCortex, VisualOutput
    from reasoning import (
        ObjectMatcher, 
        ComparisonModule, 
        TransformationSignature,
        TransformationRule,
        extract_rules,
        TransformType
    )


# ============================================================================
# TASK REPRESENTATION
# ============================================================================

@dataclass
class TrainExample:
    """A single training example."""
    input_grid: np.ndarray
    output_grid: np.ndarray


@dataclass
class Task:
    """An ARC task with training examples and test input."""
    task_id: str
    train: List[TrainExample]
    test_input: np.ndarray
    test_output: Optional[np.ndarray] = None  # For evaluation


# ============================================================================
# PHASE 3 SOLVER
# ============================================================================

class Phase3Solver:
    """
    Phase 3 Minimal Viable Solver.
    
    Pipeline:
    1. For each training example:
       - Process input/output with VisualCortex
       - Match objects
       - Detect transformation signature
    2. Extract consistent rules from all signatures
    3. Apply rules to test input
    
    NO multi-hypothesis, NO adaptation, NO stratification.
    """
    
    def __init__(self):
        self.visual_cortex = VisualCortex()
        self.matcher = ObjectMatcher()
        self.comparison = ComparisonModule()
    
    def solve(self, task: Task) -> np.ndarray:
        """
        Solve an ARC task.
        
        Returns the predicted output grid.
        """
        # Step 1: Analyze training examples
        signatures = []
        
        for example in task.train:
            sig = self._analyze_example(example)
            signatures.append(sig)
        
        # Step 2: Extract consistent rules
        rules = extract_rules(signatures)
        
        if not rules:
            # Fallback: return input unchanged
            return task.test_input.copy()
        
        # Step 3: Apply rules to test input
        result = task.test_input.copy()
        
        for rule in rules:
            result = rule.apply(result)
        
        return result
    
    def _analyze_example(self, example: TrainExample) -> TransformationSignature:
        """
        Analyze a single training example.
        """
        # Process grids
        vis_in = self.visual_cortex.process(example.input_grid)
        vis_out = self.visual_cortex.process(example.output_grid)
        
        # Match objects
        correspondence = self.matcher.match(vis_in.objects, vis_out.objects)
        
        # Detect transformation
        signature = self.comparison.compare(vis_in, vis_out, correspondence)
        
        return signature
    
    def evaluate(self, task: Task) -> Tuple[bool, float]:
        """
        Evaluate solver on a task with known test output.
        
        Returns (is_correct, accuracy) where accuracy is % matching cells.
        """
        if task.test_output is None:
            raise ValueError("Task has no test output for evaluation")
        
        prediction = self.solve(task)
        
        if prediction.shape != task.test_output.shape:
            return False, 0.0
        
        matches = (prediction == task.test_output).sum()
        total = prediction.size
        accuracy = matches / total
        
        is_correct = np.array_equal(prediction, task.test_output)
        
        return is_correct, accuracy


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def create_test_task(
    name: str,
    train_pairs: List[Tuple[List[List[int]], List[List[int]]]],
    test_input: List[List[int]],
    test_output: Optional[List[List[int]]] = None
) -> Task:
    """
    Helper to create test tasks from nested lists.
    """
    train = [
        TrainExample(
            input_grid=np.array(inp, dtype=np.int8),
            output_grid=np.array(out, dtype=np.int8)
        )
        for inp, out in train_pairs
    ]
    
    return Task(
        task_id=name,
        train=train,
        test_input=np.array(test_input, dtype=np.int8),
        test_output=np.array(test_output, dtype=np.int8) if test_output else None
    )


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_solver():
    """Test the Phase 3 solver on simple examples."""
    
    print("=" * 60)
    print("PHASE 3 SOLVER VERIFICATION")
    print("=" * 60)
    
    solver = Phase3Solver()
    
    # Test 1: Simple translation (move right by 1)
    print("\n1. Translation (move right by 1):")
    task1 = create_test_task(
        "translate_right",
        train_pairs=[
            # Example 1
            ([[0, 1, 0, 0],
              [0, 0, 0, 0]],
             [[0, 0, 1, 0],
              [0, 0, 0, 0]]),
            # Example 2
            ([[1, 0, 0, 0],
              [0, 0, 0, 0]],
             [[0, 1, 0, 0],
              [0, 0, 0, 0]]),
        ],
        test_input=[[0, 0, 1, 0],
                    [0, 0, 0, 0]],
        test_output=[[0, 0, 0, 1],
                     [0, 0, 0, 0]]
    )
    
    is_correct, acc = solver.evaluate(task1)
    print(f"   Correct: {is_correct}, Accuracy: {acc:.1%}")
    
    # Test 2: Simple recolor (blue to red)
    print("\n2. Recolor (1 -> 2):")
    task2 = create_test_task(
        "recolor",
        train_pairs=[
            ([[0, 1, 0],
              [0, 1, 0]],
             [[0, 2, 0],
              [0, 2, 0]]),
        ],
        test_input=[[1, 1, 0],
                    [0, 0, 0]],
        test_output=[[2, 2, 0],
                     [0, 0, 0]]
    )
    
    is_correct, acc = solver.evaluate(task2)
    print(f"   Correct: {is_correct}, Accuracy: {acc:.1%}")
    
    # Test 3: Identity (no change)
    print("\n3. Identity (no change):")
    task3 = create_test_task(
        "identity",
        train_pairs=[
            ([[1, 2],
              [3, 4]],
             [[1, 2],
              [3, 4]]),
        ],
        test_input=[[5, 6],
                    [7, 8]],
        test_output=[[5, 6],
                     [7, 8]]
    )
    
    is_correct, acc = solver.evaluate(task3)
    print(f"   Correct: {is_correct}, Accuracy: {acc:.1%}")
    
    print("\n" + "=" * 60)
    print("SOLVER VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    verify_solver()
