"""
Handcrafted Object Cognition Benchmark

Not randomly generated - each puzzle is carefully designed to test
specific object cognition abilities at ARC-level difficulty.

Categories:
1. Simple counting (warmup)
2. Overlapping objects (harder)
3. Partial occlusion
4. Complex shapes
5. Size variations
6. Color-based grouping
"""

import numpy as np
import json
from typing import List, Tuple, Dict


class HandcraftedObjectBenchmark:
    """Handcrafted test cases for true object cognition evaluation."""
    
    def __init__(self):
        self.puzzles = []
        self._create_all_puzzles()
    
    def _create_all_puzzles(self):
        """Create all handcrafted puzzles."""
        
        # EASY LEVEL (Should get 100%)
        self.puzzles.extend([
            self._puzzle_single_object(),
            self._puzzle_two_separate(),
            self._puzzle_three_distinct(),
            self._puzzle_empty_grid(),
        ])
        
        # MEDIUM LEVEL (Should get 80%+)
        self.puzzles.extend([
            self._puzzle_touching_objects(),
            self._puzzle_L_shapes(),
            self._puzzle_scattered_singles(),
            self._puzzle_nested_squares(),
        ])
        
        # HARD LEVEL (Target 60%+)
        self.puzzles.extend([
            self._puzzle_complex_overlapping(),
            self._puzzle_thin_connections(),
            self._puzzle_color_groups(),
            self._puzzle_partial_occlusion(),
        ])
        
        # ARC-LEVEL (Target 40%+ - very challenging!)
        self.puzzles.extend([
            self._puzzle_fractal_pattern(),
            self._puzzle_ambiguous_boundary(),
            self._puzzle_multi_color_objects(),
            self._puzzle_grid_within_grid(),
        ])
    
    # ===== EASY LEVEL =====
    
    def _puzzle_single_object(self):
        """One simple square object."""
        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        return {
            'name': 'single_square',
            'difficulty': 'easy',
            'grid': grid,
            'expected_count': 1,
            'description': 'One 2x2 square'
        }
    
    def _puzzle_two_separate(self):
        """Two clearly separate objects."""
        grid = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 3, 0],
            [0, 2, 0, 0, 3, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        return {
            'name': 'two_separate',
            'difficulty': 'easy',
            'grid': grid,
            'expected_count': 2,
            'description': 'Two vertical lines, different colors'
        }
    
    def _puzzle_three_distinct(self):
        """Three distinct objects."""
        grid = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 2, 2, 0, 3],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        return {
            'name': 'three_distinct',
            'difficulty': 'easy',
            'grid': grid,
            'expected_count': 3,
            'description': 'Single pixel, horizontal line, single pixel'
        }
    
    def _puzzle_empty_grid(self):
        """Empty grid - zero objects."""
        grid = np.zeros((4, 4), dtype=int)
        return {
            'name': 'empty',
            'difficulty': 'easy',
            'grid': grid,
            'expected_count': 0,
            'description': 'No objects'
        }
    
    # ===== MEDIUM LEVEL =====
    
    def _puzzle_touching_objects(self):
        """Objects touching but not connected."""
        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 2, 0],
            [0, 1, 1, 2, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ])
        return {
            'name': 'touching',
            'difficulty': 'medium',
            'grid': grid,
            'expected_count': 2,
            'description': 'Square touching vertical line (different colors)'
        }
    
    def _puzzle_L_shapes(self):
        """Multiple L-shaped objects."""
        grid = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 2, 0],
            [0, 1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0, 0],
        ])
        return {
            'name': 'L_shapes',
            'difficulty': 'medium',
            'grid': grid,
            'expected_count': 2,
            'description': 'Two L-shaped objects'
        }
    
    def _puzzle_scattered_singles(self):
        """Many single-pixel objects."""
        grid = np.array([
            [1, 0, 2, 0, 3],
            [0, 0, 0, 0, 0],
            [4, 0, 5, 0, 6],
        ])
        return {
            'name': 'scattered',
            'difficulty': 'medium',
            'grid': grid,
            'expected_count': 6,
            'description': 'Six scattered single pixels'
        }
    
    def _puzzle_nested_squares(self):
        """Square inside square - should be 2 objects."""
        grid = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 1, 0],
            [0, 1, 2, 2, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        return {
            'name': 'nested',
            'difficulty': 'medium',
            'grid': grid,
            'expected_count': 2,
            'description': 'Small square inside large square (different colors)'
        }
    
    # ===== HARD LEVEL =====
    
    def _puzzle_complex_overlapping(self):
        """Overlapping same-color objects."""
        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        return {
            'name': 'overlapping',
            'difficulty': 'hard',
            'grid': grid,
            'expected_count': 1,  # Connected via shared pixels
            'description': 'T-shape (connected same-color regions)'
        }
    
    def _puzzle_thin_connections(self):
        """Objects connected by thin bridges."""
        grid = np.array([
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
        ])
        return {
            'name': 'thin_bridge',
            'difficulty': 'hard',
            'grid': grid,
            'expected_count': 1,  # All connected via center bridge
            'description': 'Four squares connected by thin bridge'
        }
    
    def _puzzle_color_groups(self):
        """Multiple objects of same color scattered."""
        grid = np.array([
            [1, 0, 1, 0, 2],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 2],
        ])
        return {
            'name': 'color_groups',
            'difficulty': 'hard',
            'grid': grid,
            'expected_count': 6,  # Each pixel is separate object
            'description': 'Four 1s and two 2s (all disconnected)'
        }
    
    def _puzzle_partial_occlusion(self):
        """Simulated occlusion pattern."""
        grid = np.array([
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [1, 1, 2, 1, 1],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 0],
        ])
        return {
            'name': 'occlusion',
            'difficulty': 'hard',
            'grid': grid,
            'expected_count': 2,  # Background (1) and foreground line (2)
            'description': 'Vertical line appears to pass through square'
        }
    
    # ===== ARC LEVEL =====
    
    def _puzzle_fractal_pattern(self):
        """Hierarchical structure."""
        grid = np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
        ])
        return {
            'name': 'fractal',
            'difficulty': 'arc',
            'grid': grid,
            'expected_count': 4,  # Four separate 2x2 squares
            'description': 'Four 2x2 squares in pattern'
        }
    
    def _puzzle_ambiguous_boundary(self):
        """Where does one object end and another begin?"""
        grid = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ])
        return {
            'name': 'ambiguous',
            'difficulty': 'arc',
            'grid': grid,
            'expected_count': 2,  # Outer frame + center pixel
            'description': 'Frame with center pixel (ambiguous interpretation)'
        }
    
    def _puzzle_multi_color_objects(self):
        """Should multi-colored region be one or many objects?"""
        grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        return {
            'name': 'multi_color',
            'difficulty': 'arc',
            'grid': grid,
            'expected_count': 9,  # Each cell is different
            'description': 'Nine different colors (nine objects)'
        }
    
    def _puzzle_grid_within_grid(self):
        """Meta-structure."""
        grid = np.array([
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1],
        ])
        return {
            'name': 'grid_pattern',
            'difficulty': 'arc',
            'grid': grid,
            'expected_count': 9,  # Nine separate pixels in grid pattern
            'description': 'Grid of single pixels (pattern recognition required)'
        }
    
    def get_by_difficulty(self, difficulty: str):
        """Get all puzzles of a given difficulty."""
        return [p for p in self.puzzles if p['difficulty'] == difficulty]
    
    def save_to_json(self, filepath: str):
        """Save benchmarks to JSON."""
        # Convert numpy arrays to lists for JSON
        export_data = []
        for puzzle in self.puzzles:
            export_data.append({
                'name': puzzle['name'],
                'difficulty': puzzle['difficulty'],
                'grid': puzzle['grid'].tolist(),
                'expected_count': puzzle['expected_count'],
                'description': puzzle['description']
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        return self.puzzles[idx]


def create_handcrafted_benchmark():
    """Create and save the handcrafted benchmark."""
    benchmark = HandcraftedObjectBenchmark()
    
    print("="*70)
    print("HANDCRAFTED OBJECT COGNITION BENCHMARK")
    print("="*70)
    print(f"\nTotal puzzles: {len(benchmark)}")
    print(f"\nBy difficulty:")
    for diff in ['easy', 'medium', 'hard', 'arc']:
        count = len(benchmark.get_by_difficulty(diff))
        print(f"  {diff.upper():8s}: {count} puzzles")
    
    # Save to file
    filepath = "data/handcrafted_object_benchmark.json"
    benchmark.save_to_json(filepath)
    print(f"\n✓ Saved to {filepath}")
    
    # Show samples
    print(f"\n{'='*70}")
    print("SAMPLE PUZZLES")
    print(f"{'='*70}\n")
    
    for diff in ['easy', 'medium', 'hard', 'arc']:
        puzzles = benchmark.get_by_difficulty(diff)
        if puzzles:
            puzzle = puzzles[0]
            print(f"\n📋 {diff.upper()}: {puzzle['name']}")
            print(f"   {puzzle['description']}")
            print(f"   Expected: {puzzle['expected_count']} objects")
            print(f"   Grid:")
            for row in puzzle['grid']:
                print(f"     {' '.join(str(x) for x in row)}")
    
    return benchmark


if __name__ == "__main__":
    benchmark = create_handcrafted_benchmark()
    
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA")
    print(f"{'='*70}\n")
    print("For model to be ARC-ready:")
    print("  EASY level:   100% accuracy (basic object detection)")
    print("  MEDIUM level:  80% accuracy (complex shapes)")
    print("  HARD level:    60% accuracy (ambiguous cases)")
    print("  ARC level:     40% accuracy (meta-reasoning)")
    print("\nOverall: 70%+ across all levels → STRONG TEACHER ✓")
