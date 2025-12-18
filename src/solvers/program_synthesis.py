"""Program synthesis solver for ARC-AGI."""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict

from .base import Solver, SolverResult
from ..core.grid import Grid
from ..core.task import Task, TaskPair
from ..core.primitives import Primitive, PrimitiveLibrary


@dataclass
class Program:
    """
    A synthesized program that transforms grids.
    
    Programs are compositions of primitives.
    """
    
    expression: str  # Human-readable representation
    execute: Callable[[Grid], Grid]  # Executable function
    cost: float = 1.0  # Complexity cost (used in search)
    
    def __call__(self, grid: Grid) -> Grid:
        """Execute the program on a grid."""
        return self.execute(grid)
    
    def __repr__(self) -> str:
        return f"Program({self.expression})"
    
    def __lt__(self, other):
        """For priority queue sorting."""
        return self.cost < other.cost


class ProgramSynthesisSolver(Solver):
    """
    Solver that synthesizes programs from examples.
    
    This implements a bottom-up enumerative synthesis approach.
    """
    
    def __init__(
        self,
        max_depth: int = 5,
        beam_width: int = 100,
        timeout: float = 60.0,
        name: str = "program_synthesis"
    ):
        super().__init__(name=name)
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.timeout = timeout
        self.primitives = PrimitiveLibrary()
    
    def _generate_base_programs(self) -> List[Program]:
        """Generate base-level programs (depth 1)."""
        programs = []
        
        # Identity
        programs.append(Program(
            expression="identity",
            execute=lambda g: g.copy(),
            cost=0.1
        ))
        
        # Geometric transforms
        programs.append(Program(
            expression="rotate_90",
            execute=lambda g: g.rotate_90(),
            cost=1.0
        ))
        programs.append(Program(
            expression="rotate_180",
            execute=lambda g: g.rotate_180(),
            cost=1.0
        ))
        programs.append(Program(
            expression="rotate_270",
            execute=lambda g: g.rotate_270(),
            cost=1.0
        ))
        programs.append(Program(
            expression="flip_h",
            execute=lambda g: g.flip_horizontal(),
            cost=1.0
        ))
        programs.append(Program(
            expression="flip_v",
            execute=lambda g: g.flip_vertical(),
            cost=1.0
        ))
        programs.append(Program(
            expression="transpose",
            execute=lambda g: g.transpose(),
            cost=1.0
        ))
        
        # Crop to content
        programs.append(Program(
            expression="crop_content",
            execute=lambda g: g.crop_to_content() or g,
            cost=1.5
        ))
        
        # Scale operations
        for factor in [2, 3]:
            programs.append(Program(
                expression=f"scale_{factor}",
                execute=lambda g, f=factor: g.scale(f),
                cost=1.5
            ))
        
        # Color replacements (limited to save search space)
        for from_c in range(5):
            for to_c in range(5):
                if from_c != to_c:
                    programs.append(Program(
                        expression=f"replace_{from_c}_with_{to_c}",
                        execute=lambda g, f=from_c, t=to_c: g.replace_color(f, t),
                        cost=2.0
                    ))
        
        return programs
    
    def _compose_programs(
        self,
        prog1: Program,
        prog2: Program
    ) -> Program:
        """Compose two programs: prog2(prog1(x))."""
        return Program(
            expression=f"{prog2.expression}({prog1.expression})",
            execute=lambda g, p1=prog1, p2=prog2: p2(p1(g)),
            cost=prog1.cost + prog2.cost
        )
    
    def _evaluate_program(
        self,
        program: Program,
        pairs: List[TaskPair]
    ) -> Tuple[bool, float]:
        """
        Evaluate if a program works for all pairs.
        
        Returns (matches_all, similarity_score)
        """
        total_similarity = 0.0
        matches_all = True
        
        for pair in pairs:
            try:
                result = program(pair.input)
                
                if result == pair.output:
                    total_similarity += 1.0
                else:
                    matches_all = False
                    total_similarity += result.similarity(pair.output)
                    
            except Exception:
                matches_all = False
                total_similarity += 0.0
        
        avg_similarity = total_similarity / len(pairs) if pairs else 0.0
        return matches_all, avg_similarity
    
    def _synthesize(self, task: Task) -> Optional[Program]:
        """
        Synthesize a program that solves the task.
        
        Uses beam search over the program space.
        """
        import time
        start_time = time.time()
        
        # Start with base programs
        programs = self._generate_base_programs()
        
        # Track best programs by similarity
        best_programs = []  # (similarity, program) heap
        
        # Evaluate base programs
        for prog in programs:
            matches, similarity = self._evaluate_program(prog, task.train)
            
            if matches:
                return prog  # Found perfect match
            
            heapq.heappush(best_programs, (-similarity, prog))
        
        # Keep only beam_width best
        best_programs = heapq.nsmallest(self.beam_width, best_programs)
        
        # Iteratively compose programs
        base_programs = programs[:10]  # Use simpler ones for composition
        
        for depth in range(2, self.max_depth + 1):
            if time.time() - start_time > self.timeout:
                break
            
            new_programs = []
            
            for _, prog in best_programs:
                for base in base_programs:
                    # Try composing in both orders
                    for composed in [
                        self._compose_programs(prog, base),
                        self._compose_programs(base, prog)
                    ]:
                        matches, similarity = self._evaluate_program(
                            composed, task.train
                        )
                        
                        if matches:
                            return composed
                        
                        if similarity > 0.3:  # Only keep promising ones
                            new_programs.append((-similarity, composed))
            
            # Update beam
            best_programs = heapq.nsmallest(
                self.beam_width,
                best_programs + new_programs
            )
        
        # Return best non-perfect match if any
        if best_programs:
            return best_programs[0][1]
        
        return None
    
    def solve(self, task: Task) -> SolverResult:
        """Solve the task using program synthesis."""
        
        # Try to synthesize a program
        program = self._synthesize(task)
        
        predictions = []
        confidence = []
        
        if program is not None:
            # Evaluate quality on training data
            matches, similarity = self._evaluate_program(program, task.train)
            conf = 0.95 if matches else similarity
            
            for test_pair in task.test:
                try:
                    pred = program(test_pair.input)
                    predictions.append([pred])
                    confidence.append([conf])
                except Exception:
                    predictions.append([test_pair.input.copy()])
                    confidence.append([0.1])
        else:
            # No program found
            for test_pair in task.test:
                predictions.append([test_pair.input.copy()])
                confidence.append([0.1])
        
        return SolverResult(
            task_id=task.task_id,
            predictions=predictions,
            confidence=confidence,
            metadata={
                "program": str(program) if program else None,
                "synthesized": program is not None
            }
        )


class DSLSearcher:
    """
    Domain-specific language searcher for ARC.
    
    Implements more sophisticated search strategies.
    """
    
    def __init__(self, dsl: PrimitiveLibrary):
        self.dsl = dsl
        self.cache = {}
    
    def search_with_abstraction(
        self,
        task: Task,
        max_iterations: int = 1000
    ) -> Optional[Program]:
        """
        Search with learned abstractions.
        
        This is a more advanced technique that:
        1. Identifies common sub-patterns
        2. Creates new abstractions from them
        3. Uses abstractions in further search
        """
        # TODO: Implement abstraction learning
        # This is a key research direction for ARC
        pass
    
    def search_with_anti_unification(
        self,
        task: Task
    ) -> Optional[Program]:
        """
        Use anti-unification to find common structure.
        
        Anti-unification finds the most specific generalization
        of multiple examples.
        """
        # TODO: Implement anti-unification
        pass
