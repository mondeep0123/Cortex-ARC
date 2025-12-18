"""Task representation for ARC-AGI."""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from .grid import Grid


@dataclass
class TaskPair:
    """A single input-output pair in an ARC task."""
    
    input: Grid
    output: Grid
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskPair:
        """Create a TaskPair from a dictionary."""
        return cls(
            input=Grid.from_list(data["input"]),
            output=Grid.from_list(data["output"])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "input": self.input.to_list(),
            "output": self.output.to_list()
        }
    
    @property
    def input_shape(self) -> tuple:
        """Return the shape of the input grid."""
        return self.input.shape
    
    @property
    def output_shape(self) -> tuple:
        """Return the shape of the output grid."""
        return self.output.shape
    
    @property
    def shape_changed(self) -> bool:
        """Check if the shape changed from input to output."""
        return self.input_shape != self.output_shape
    
    def __repr__(self) -> str:
        return f"TaskPair(input={self.input.shape}, output={self.output.shape})"


@dataclass
class Task:
    """
    Represents a complete ARC-AGI task.
    
    A task consists of:
    - train: A list of input-output pairs showing the transformation
    - test: A list of input-output pairs to predict
    
    The goal is to learn the transformation from training pairs
    and apply it to test inputs.
    """
    
    task_id: str
    train: List[TaskPair]
    test: List[TaskPair]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], task_id: str = "unknown") -> Task:
        """Create a Task from a dictionary."""
        train_pairs = [TaskPair.from_dict(p) for p in data["train"]]
        test_pairs = [TaskPair.from_dict(p) for p in data["test"]]
        
        return cls(
            task_id=task_id,
            train=train_pairs,
            test=test_pairs
        )
    
    @classmethod
    def from_json(cls, path: Path) -> Task:
        """Load a Task from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        task_id = path.stem  # Use filename as task ID
        return cls.from_dict(data, task_id=task_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "train": [p.to_dict() for p in self.train],
            "test": [p.to_dict() for p in self.test]
        }
    
    def to_json(self, path: Path):
        """Save the task to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @property
    def num_train(self) -> int:
        """Number of training pairs."""
        return len(self.train)
    
    @property
    def num_test(self) -> int:
        """Number of test pairs."""
        return len(self.test)
    
    @property
    def train_inputs(self) -> List[Grid]:
        """Get all training input grids."""
        return [p.input for p in self.train]
    
    @property
    def train_outputs(self) -> List[Grid]:
        """Get all training output grids."""
        return [p.output for p in self.train]
    
    @property
    def test_inputs(self) -> List[Grid]:
        """Get all test input grids."""
        return [p.input for p in self.test]
    
    @property
    def test_outputs(self) -> List[Grid]:
        """Get all test output grids (ground truth)."""
        return [p.output for p in self.test]
    
    # =========== Analysis Methods ===========
    
    def all_colors(self) -> set:
        """Get all colors used in the task."""
        colors = set()
        for pair in self.train + self.test:
            colors.update(pair.input.unique_colors())
            colors.update(pair.output.unique_colors())
        return colors
    
    def shapes_consistent(self) -> bool:
        """Check if input/output shapes are consistent across pairs."""
        if not self.train:
            return True
        
        first_input_shape = self.train[0].input_shape
        first_output_shape = self.train[0].output_shape
        
        for pair in self.train[1:]:
            if pair.input_shape != first_input_shape:
                return False
            if pair.output_shape != first_output_shape:
                return False
        
        return True
    
    def shape_relationship(self) -> str:
        """Analyze the relationship between input and output shapes."""
        relationships = []
        
        for pair in self.train:
            ih, iw = pair.input_shape
            oh, ow = pair.output_shape
            
            if (ih, iw) == (oh, ow):
                relationships.append("same")
            elif oh == ih and ow == iw:
                relationships.append("same")
            elif oh > ih or ow > iw:
                relationships.append("larger")
            elif oh < ih or ow < iw:
                relationships.append("smaller")
            else:
                relationships.append("mixed")
        
        # Check consistency
        if len(set(relationships)) == 1:
            return relationships[0]
        return "variable"
    
    def input_size_range(self) -> tuple:
        """Get the range of input sizes."""
        sizes = [p.input.size for p in self.train + self.test]
        return (min(sizes), max(sizes))
    
    def output_size_range(self) -> tuple:
        """Get the range of output sizes."""
        sizes = [p.output.size for p in self.train + self.test]
        return (min(sizes), max(sizes))
    
    def difficulty_estimate(self) -> float:
        """
        Estimate task difficulty based on heuristics.
        Returns a score from 0 (easy) to 1 (hard).
        """
        score = 0.0
        
        # More colors = harder
        num_colors = len(self.all_colors())
        score += min(num_colors / 10, 0.3)
        
        # Shape changes = harder
        if not self.shapes_consistent():
            score += 0.2
        
        relationship = self.shape_relationship()
        if relationship == "variable":
            score += 0.2
        
        # Larger grids = harder
        _, max_size = self.input_size_range()
        score += min(max_size / 900, 0.3)  # 900 = 30x30
        
        return min(score, 1.0)
    
    def __repr__(self) -> str:
        return f"Task(id={self.task_id}, train={self.num_train}, test={self.num_test})"
    
    def summary(self) -> str:
        """Generate a summary of the task."""
        lines = [
            f"Task ID: {self.task_id}",
            f"Training pairs: {self.num_train}",
            f"Test pairs: {self.num_test}",
            f"Colors used: {sorted(self.all_colors())}",
            f"Shape relationship: {self.shape_relationship()}",
            f"Difficulty estimate: {self.difficulty_estimate():.2f}",
            "",
            "Training shapes:",
        ]
        
        for i, pair in enumerate(self.train):
            lines.append(f"  {i+1}. {pair.input_shape} -> {pair.output_shape}")
        
        lines.append("\nTest shapes:")
        for i, pair in enumerate(self.test):
            lines.append(f"  {i+1}. {pair.input_shape} -> ?")
        
        return "\n".join(lines)
