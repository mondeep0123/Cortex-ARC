"""Evaluation module for ARC-AGI solutions."""

from .metrics import accuracy, exact_match, partial_match, evaluate_task
from .evaluator import Evaluator
from .submission import generate_submission, validate_submission

__all__ = [
    "accuracy",
    "exact_match",
    "partial_match",
    "evaluate_task",
    "Evaluator",
    "generate_submission",
    "validate_submission",
]
