"""Base classes for curriculum skill modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

# Type alias for grids
Grid = np.ndarray


@dataclass
class Task:
    """Represents a single task (training or test example)."""
    train_pairs: List[tuple[Grid, Grid]]  # List of (input, output) pairs
    test_inputs: List[Grid]  # Test inputs (outputs unknown during solving)
    test_outputs: Optional[List[Grid]] = None  # Ground truth (for evaluation only)
    metadata: Dict[str, Any] = None  # Additional task information
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SkillOutput:
    """Output from a skill module."""
    result: Grid  # Resulting grid after applying skill
    confidence: float  # Confidence score (0.0 to 1.0)
    reasoning: str  # Human-readable explanation of what was done
    intermediate_steps: Optional[List[Grid]] = None  # Steps taken (for visualization)
    
    def __post_init__(self):
        if self.intermediate_steps is None:
            self.intermediate_steps = []


@dataclass
class SkillMetrics:
    """Metrics for evaluating skill performance."""
    accuracy: float  # Fraction of tasks solved correctly
    avg_confidence: float  # Average confidence on correct solutions
    avg_attempts: float  # Average number of attempts needed
    generalization_score: float  # Performance on novel variations
    
    def __str__(self):
        return (f"Accuracy: {self.accuracy:.2%}, "
                f"Confidence: {self.avg_confidence:.2f}, "
                f"Attempts: {self.avg_attempts:.1f}, "
                f"Generalization: {self.generalization_score:.2%}")


class SkillModule(ABC):
    """
    Abstract base class for all curriculum skills.
    
    Each skill module can:
    1. Apply the skill to transform a grid
    2. Train on curriculum-specific tasks
    3. Evaluate performance on held-out tasks
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.training_history = []
    
    @abstractmethod
    def apply(self, grid: Grid, context: Optional[Dict[str, Any]] = None) -> SkillOutput:
        """
        Apply this skill to a grid.
        
        Args:
            grid: Input grid
            context: Optional context information (e.g., from demonstration pairs)
        
        Returns:
            SkillOutput containing result and metadata
        """
        pass
    
    @abstractmethod
    def can_apply(self, task: Task) -> float:
        """
        Determine if this skill is applicable to a task.
        
        Args:
            task: Task to analyze
        
        Returns:
            Confidence score (0.0 to 1.0) that this skill is relevant
        """
        pass
    
    def train_on_task(self, task: Task) -> Dict[str, float]:
        """
        Train the skill module on a single task.
        
        Args:
            task: Training task
        
        Returns:
            Dictionary of training metrics
        """
        # Default implementation - can be overridden
        return {"loss": 0.0}
    
    def evaluate(self, tasks: List[Task]) -> SkillMetrics:
        """
        Evaluate skill performance on a set of tasks.
        
        Args:
            tasks: List of evaluation tasks
        
        Returns:
            SkillMetrics summarizing performance
        """
        correct = 0
        total_confidence = 0.0
        total_attempts = 0
        
        for task in tasks:
            # Try to solve each test input
            for test_idx, test_input in enumerate(task.test_inputs):
                output = self.apply(test_input)
                
                # Check if correct (if ground truth available)
                if task.test_outputs is not None:
                    expected = task.test_outputs[test_idx]
                    if np.array_equal(output.result, expected):
                        correct += 1
                        total_confidence += output.confidence
                
                total_attempts += 1
        
        accuracy = correct / total_attempts if total_attempts > 0 else 0.0
        avg_confidence = total_confidence / correct if correct > 0 else 0.0
        
        return SkillMetrics(
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            avg_attempts=1.0,  # Placeholder
            generalization_score=accuracy  # Placeholder
        )
    
    def save(self, path: str):
        """Save skill module state."""
        # To be implemented based on specific module needs
        pass
    
    def load(self, path: str):
        """Load skill module state."""
        # To be implemented based on specific module needs
        pass
    
    def __str__(self):
        status = "trained" if self.is_trained else "untrained"
        return f"{self.name} ({status})"
    
    def __repr__(self):
        return f"SkillModule(name='{self.name}', trained={self.is_trained})"


class CompositeSkill(SkillModule):
    """
    A skill composed of multiple sub-skills applied in sequence.
    
    This allows building complex behaviors from simpler primitives.
    """
    
    def __init__(self, name: str, skills: List[SkillModule]):
        super().__init__(name)
        self.skills = skills
    
    def apply(self, grid: Grid, context: Optional[Dict[str, Any]] = None) -> SkillOutput:
        """Apply all sub-skills in sequence."""
        current = grid
        all_steps = [grid]
        combined_reasoning = []
        min_confidence = 1.0
        
        for skill in self.skills:
            output = skill.apply(current, context)
            current = output.result
            all_steps.append(current)
            combined_reasoning.append(f"{skill.name}: {output.reasoning}")
            min_confidence = min(min_confidence, output.confidence)
        
        return SkillOutput(
            result=current,
            confidence=min_confidence,
            reasoning=" → ".join(combined_reasoning),
            intermediate_steps=all_steps
        )
    
    def can_apply(self, task: Task) -> float:
        """Average confidence across all sub-skills."""
        confidences = [skill.can_apply(task) for skill in self.skills]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def add_skill(self, skill: SkillModule):
        """Add a skill to the composition."""
        self.skills.append(skill)
    
    def __str__(self):
        skill_names = " → ".join([s.name for s in self.skills])
        return f"{self.name} [{skill_names}]"


class SkillLibrary:
    """
    Registry of all available skills.
    
    Manages skill modules and helps select appropriate skills for tasks.
    """
    
    def __init__(self):
        self.skills: Dict[str, SkillModule] = {}
        self.skill_categories: Dict[str, List[str]] = {
            "core": [],
            "operations": [],
            "meta": []
        }
    
    def register(self, skill: SkillModule, category: str = "operations"):
        """Register a skill in the library."""
        self.skills[skill.name] = skill
        if category in self.skill_categories:
            self.skill_categories[category].append(skill.name)
    
    def get_skill(self, name: str) -> Optional[SkillModule]:
        """Get a skill by name."""
        return self.skills.get(name)
    
    def get_relevant_skills(self, task: Task, threshold: float = 0.3) -> List[SkillModule]:
        """
        Find skills relevant to a task.
        
        Args:
            task: Task to analyze
            threshold: Minimum confidence threshold
        
        Returns:
            List of relevant skills sorted by confidence
        """
        relevant = []
        
        for skill in self.skills.values():
            confidence = skill.can_apply(task)
            if confidence >= threshold:
                relevant.append((skill, confidence))
        
        # Sort by confidence (descending)
        relevant.sort(key=lambda x: x[1], reverse=True)
        
        return [skill for skill, _ in relevant]
    
    def list_skills(self, category: Optional[str] = None) -> List[str]:
        """List all skills, optionally filtered by category."""
        if category and category in self.skill_categories:
            return self.skill_categories[category]
        return list(self.skills.keys())
    
    def __len__(self):
        return len(self.skills)
    
    def __str__(self):
        return f"SkillLibrary({len(self.skills)} skills)"
