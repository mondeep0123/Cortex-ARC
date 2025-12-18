"""Solvers module - Various approaches to solving ARC-AGI tasks."""

from .base import Solver, SolverResult
from .brute_force import BruteForceSolver
from .program_synthesis import ProgramSynthesisSolver

__all__ = [
    "Solver",
    "SolverResult",
    "BruteForceSolver",
    "ProgramSynthesisSolver",
]
