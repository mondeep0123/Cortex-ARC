"""
Pure Numerosity Benchmark

Tests ONLY counting abilities - no segmentation needed!

Task Types:
1. Total counting: How many non-background cells?
2. Color counting: How many of color X?
3. Max color: Which color appears most?

Success Metric: Exact count (or within ±1 for harder cases)
"""

import numpy as np
import json
from typing import List, Dict


class PureNumerosityBenchmark:
    """Tests pure counting without other primitives."""
    
    def __init__(self):
        self.puzzles = []
        self._create_all_puzzles()
    
    def _create_all_puzzles(self):
        """Create all numerosity puzzles."""
        
        # EASY LEVEL - Simple counting
        self.puzzles.extend([
            self._puzzle_empty(),
            self._puzzle_single(),
            self._puzzle_small_count(),
            self._puzzle_exact_ten(),
        ])
        
        # MEDIUM LEVEL - Color counting
        self.puzzles.extend([
            self._puzzle_single_color(),
            self._puzzle_two_colors(),
            self._puzzle_dominant_color(),
            self._puzzle_equal_colors(),
        ])
        
        # HARD LEVEL - Complex counting
        self.puzzles.extend([
            self._puzzle_many_colors(),
            self._puzzle_sparse(),
            self._puzzle_dense(),
            self._puzzle_pattern_count(),
        ])
        
        # ARC LEVEL - Challenging
        self.puzzles.extend([
            self._puzzle_large_count(),
            self._puzzle_rare_color(),
            self._puzzle_max_ambiguous(),
            self._puzzle_full_grid(),
        ])
    
    # ===== EASY LEVEL =====
    
    def _puzzle_empty(self):
        """Empty grid - count 0."""
        grid = np.zeros((3, 3), dtype=np.int32)
        
        return {
            'name': 'empty',
            'difficulty': 'easy',
            'grid': grid,
            'expected_total': 0,
            'expected_color_counts': [9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'expected_max_color': 0,  # Background
            'description': 'Empty grid - should count 0'
        }
    
    def _puzzle_single(self):
        """Single cell."""
        grid = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.int32)
        
        color_counts = [8, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        
        return {
            'name': 'single',
            'difficulty': 'easy',
            'grid': grid,
            'expected_total': 1,
            'expected_color_counts': color_counts,
            'expected_max_color': 1,
            'description': 'Single colored cell'
        }
    
    def _puzzle_small_count(self):
        """Count 5 cells."""
        grid = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ], dtype=np.int32)
        
        color_counts = [4, 5, 0, 0, 0, 0, 0, 0, 0, 0]
        
        return {
            'name': 'small_count',
            'difficulty': 'easy',
            'grid': grid,
            'expected_total': 5,
            'expected_color_counts': color_counts,
            'expected_max_color': 1,
            'description': 'Count 5 cells in cross pattern'
        }
    
    def _puzzle_exact_ten(self):
        """Count exactly 10."""
        grid = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ], dtype=np.int32)
        
        color_counts = [5, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        
        return {
            'name': 'exact_ten',
            'difficulty': 'easy',
            'grid': grid,
            'expected_total': 10,
            'expected_color_counts': color_counts,
            'expected_max_color': 1,
            'description': 'Count exactly 10 cells'
        }
    
    # ===== MEDIUM LEVEL =====
    
    def _puzzle_single_color(self):
        """All same color."""
        grid = np.array([
            [2, 2, 0],
            [2, 2, 2],
            [0, 2, 2]
        ], dtype=np.int32)
        
        color_counts = [2, 0, 7, 0, 0, 0, 0, 0, 0, 0]
        
        return {
            'name': 'single_color_many',
            'difficulty': 'medium',
            'grid': grid,
            'expected_total': 7,
            'expected_color_counts': color_counts,
            'expected_max_color': 2,
            'description': 'All colored cells are same color'
        }
    
    def _puzzle_two_colors(self):
        """Two different colors."""
        grid = np.array([
            [1, 1, 1, 0],
            [2, 2, 2, 2],
        ], dtype=np.int32)
        
        color_counts = [1, 3, 4, 0, 0, 0, 0, 0, 0, 0]
        
        return {
            'name': 'two_colors',
            'difficulty': 'medium',
            'grid': grid,
            'expected_total': 7,
            'expected_color_counts': color_counts,
            'expected_max_color': 2,  # 2 appears 4 times
            'description': 'Two colors, find which has more'
        }
    
    def _puzzle_dominant_color(self):
        """One color dominates."""
        grid = np.array([
            [3, 3, 3, 3],
            [3, 5, 5, 3],
            [3, 3, 3, 3]
        ], dtype=np.int32)
        
        color_counts = [0, 0, 0, 10, 0, 2, 0, 0, 0, 0]
        
        return {
            'name': 'dominant_color',
            'difficulty': 'medium',
            'grid': grid,
            'expected_total': 12,
            'expected_color_counts': color_counts,
            'expected_max_color': 3,  # 3 appears 10 times
            'description': 'One color dominates (10 vs 2)'
        }
    
    def _puzzle_equal_colors(self):
        """Equal count of two colors."""
        grid = np.array([
            [4, 4, 4],
            [0, 0, 0],
            [7, 7, 7]
        ], dtype=np.int32)
        
        color_counts = [3, 0, 0, 0, 3, 0, 0, 3, 0, 0]
        
        return {
            'name': 'equal_colors',
            'difficulty': 'medium',
            'grid': grid,
            'expected_total': 6,
            'expected_color_counts': color_counts,
            'expected_max_color': 4,  # Either 4 or 7 (tied at 3 each)
            'description': 'Two colors with equal counts'
        }
    
    # ===== HARD LEVEL =====
    
    def _puzzle_many_colors(self):
        """Many different colors."""
        grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.int32)
        
        color_counts = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        return {
            'name': 'many_colors',
            'difficulty': 'hard',
            'grid': grid,
            'expected_total': 9,
            'expected_color_counts': color_counts,
            'expected_max_color': 1,  # All tied at 1, return first
            'description': 'All 9 colors, each appears once'
        }
    
    def _puzzle_sparse(self):
        """Very sparse - few colored cells."""
        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.int32)
        
        color_counts = [23, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        
        return {
            'name': 'sparse',
            'difficulty': 'hard',
            'grid': grid,
            'expected_total': 2,
            'expected_color_counts': color_counts,
            'expected_max_color': 1,  # Both 1 and 2 tied, return lower
            'description': 'Very sparse grid (2 out of 25)'
        }
    
    def _puzzle_dense(self):
        """Very dense - mostly colored."""
        grid = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ], dtype=np.int32)
        
        color_counts = [2, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        
        return {
            'name': 'dense',
            'difficulty': 'hard',
            'grid': grid,
            'expected_total': 10,
            'expected_color_counts': color_counts,
            'expected_max_color': 1,
            'description': 'Very dense grid (10 out of 12)'
        }
    
    def _puzzle_pattern_count(self):
        """Repeating pattern."""
        grid = np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1]
        ], dtype=np.int32)
        
        color_counts = [7, 8, 0, 0, 0, 0, 0, 0, 0, 0]
        
        return {
            'name': 'checkerboard',
            'difficulty': 'hard',
            'grid': grid,
            'expected_total': 8,
            'expected_color_counts': color_counts,
            'expected_max_color': 1,
            'description': 'Checkerboard pattern - count colored cells'
        }
    
    # ===== ARC LEVEL =====
    
    def _puzzle_large_count(self):
        """Large count (20+)."""
        grid = np.array([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [0, 0, 0, 0, 5]
        ], dtype=np.int32)
        
        color_counts = [4, 0, 0, 0, 0, 21, 0, 0, 0, 0]
        
        return {
            'name': 'large_count',
            'difficulty': 'arc',
            'grid': grid,
            'expected_total': 21,
            'expected_color_counts': color_counts,
            'expected_max_color': 5,
            'description': 'Large count (21 cells)',
            'tolerance': 1  # Allow ±1 for large counts
        }
    
    def _puzzle_rare_color(self):
        """Rare color among many."""
        grid = np.array([
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 9, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ], dtype=np.int32)
        
        color_counts = [0, 0, 17, 0, 0, 0, 0, 0, 0, 1]
        
        return {
            'name': 'rare_color',
            'difficulty': 'arc',
            'grid': grid,
            'expected_total': 18,
            'expected_color_counts': color_counts,
            'expected_max_color': 2,  # 2 dominates with 17
            'description': 'Find rare color (1) among dominant (17)'
        }
    
    def _puzzle_max_ambiguous(self):
        """Maximum color is ambiguous (near tie)."""
        grid = np.array([
            [3, 3, 3, 3, 3],
            [6, 6, 6, 6, 0]
        ], dtype=np.int32)
        
        color_counts = [1, 0, 0, 5, 0, 0, 4, 0, 0, 0]
        
        return {
            'name': 'close_tie',
            'difficulty': 'arc',
            'grid': grid,
            'expected_total': 9,
            'expected_color_counts': color_counts,
            'expected_max_color': 3,  # 3 has 5, 6 has 4
            'description': 'Close tie: 5 vs 4',
            'tolerance': 0  # Must be exact for max color
        }
    
    def _puzzle_full_grid(self):
        """Completely filled grid."""
        grid = np.array([
            [8, 8, 8],
            [8, 8, 8],
            [8, 8, 8]
        ], dtype=np.int32)
        
        color_counts = [0, 0, 0, 0, 0, 0, 0, 0, 9, 0]
        
        return {
            'name': 'full_grid',
            'difficulty': 'arc',
            'grid': grid,
            'expected_total': 9,
            'expected_color_counts': color_counts,
            'expected_max_color': 8,
            'description': 'Completely filled grid (all one color)'
        }
    
    def get_by_difficulty(self, difficulty: str):
        """Get all puzzles of a given difficulty."""
        return [p for p in self.puzzles if p['difficulty'] == difficulty]
    
    def save_to_json(self, filepath: str):
        """Save benchmarks to JSON."""
        export_data = []
        for puzzle in self.puzzles:
            export_data.append({
                'name': puzzle['name'],
                'difficulty': puzzle['difficulty'],
                'grid': puzzle['grid'].tolist(),
                'expected_total': int(puzzle['expected_total']),
                'expected_color_counts': [int(x) for x in puzzle['expected_color_counts']],
                'expected_max_color': int(puzzle['expected_max_color']),
                'description': puzzle['description'],
                'tolerance': puzzle.get('tolerance', 0)
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        return self.puzzles[idx]


if __name__ == "__main__":
    benchmark = PureNumerosityBenchmark()
    
    print("="*70)
    print("PURE NUMEROSITY BENCHMARK")
    print("="*70)
    print(f"\nTotal puzzles: {len(benchmark)}")
    print(f"\nBy difficulty:")
    for diff in ['easy', 'medium', 'hard', 'arc']:
        count = len(benchmark.get_by_difficulty(diff))
        print(f"  {diff.upper():8s}: {count} puzzles")
    
    # Save
    filepath = "data/pure_numerosity_benchmark.json"
    benchmark.save_to_json(filepath)
    print(f"\n✓ Saved to {filepath}")
    
    # Show samples
    print(f"\n{'='*70}")
    print("TASK: Count colored cells (exact count)")
    print(f"{'='*70}\n")
    
    for diff in ['easy', 'medium', 'hard', 'arc']:
        puzzles = benchmark.get_by_difficulty(diff)
        if puzzles:
            puzzle = puzzles[0]
            print(f"\n📋 {diff.upper()}: {puzzle['name']}")
            print(f"   {puzzle['description']}")
            print(f"   Expected total: {puzzle['expected_total']}")
            print(f"   Expected max color: {puzzle['expected_max_color']}")
            print(f"   Grid:")
            for row in puzzle['grid']:
                print(f"     {' '.join(str(x) for x in row)}")
    
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA")
    print(f"{'='*70}\n")
    print("Evaluation metric: Exact count (or ±1 for large counts)")
    print("  EASY level:   100% accuracy (simple counts)")
    print("  MEDIUM level:  90% accuracy (color-specific)")
    print("  HARD level:    80% accuracy (complex patterns)")
    print("  ARC level:     70% accuracy (challenging)")
    print("\nOverall: 85%+ across all levels → STRONG PRIMITIVE ✓")
    print("\nThis tests ONLY counting - no segmentation needed!")
