"""
Pure Object Cognition Benchmark

Tests ONLY object cognition abilities:
- Segmentation (separate objects from background)
- Boundary detection
- Connectivity understanding
- Object persistence

NO counting, NO geometry, NO other primitives!

Task: Given a grid, predict which cells belong to objects vs background
Success: IoU (Intersection over Union) > threshold
"""

import numpy as np
import json
from typing import List, Tuple, Dict


class PureObjectCognitionBenchmark:
    """Tests pure object detection/segmentation without numerosity."""
    
    def __init__(self):
        self.puzzles = []
        self._create_all_puzzles()
    
    def _create_all_puzzles(self):
        """Create all pure object cognition puzzles."""
        
        # EASY LEVEL - Clear object-background separation
        self.puzzles.extend([
            self._puzzle_single_clear_object(),
            self._puzzle_two_separated_objects(),
            self._puzzle_straight_line(),
            self._puzzle_empty_vs_filled(),
        ])
        
        # MEDIUM LEVEL - Same-color connectivity
        self.puzzles.extend([
            self._puzzle_connected_vs_separate(),
            self._puzzle_L_shape_connectivity(),
            self._puzzle_diagonal_not_connected(),
            self._puzzle_multiple_same_color(),
        ])
        
        # HARD LEVEL - Complex shapes and connectivity
        self.puzzles.extend([
            self._puzzle_complex_connected_shape(),
            self._puzzle_touching_different_colors(),
            self._puzzle_nested_different_colors(),
            self._puzzle_partial_overlap(),
        ])
        
        # ARC LEVEL - Ambiguous cases
        self.puzzles.extend([
            self._puzzle_thin_connection(),
            self._puzzle_surrounding_frame(),
            self._puzzle_diagonal_proximity(),
            self._puzzle_color_boundaries(),
        ])
    
    # ===== EASY LEVEL =====
    
    def _puzzle_single_clear_object(self):
        """One object, clear background."""
        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        # Expected: cells with value 1 are object, 0 is background
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'single_clear_object',
            'difficulty': 'easy',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Segment single 2x2 square from background',
            'success_iou': 1.0  # Perfect segmentation expected
        }
    
    def _puzzle_two_separated_objects(self):
        """Two objects, well separated."""
        grid = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 3, 0],
            [0, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'two_separated',
            'difficulty': 'easy',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect two separated colored regions',
            'success_iou': 1.0
        }
    
    def _puzzle_straight_line(self):
        """Horizontal line object."""
        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'straight_line',
            'difficulty': 'easy',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect horizontal line as single object',
            'success_iou': 1.0
        }
    
    def _puzzle_empty_vs_filled(self):
        """Some filled, some empty."""
        grid = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'sparse_objects',
            'difficulty': 'easy',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect sparse objects vs background',
            'success_iou': 1.0
        }
    
    # ===== MEDIUM LEVEL =====
    
    def _puzzle_connected_vs_separate(self):
        """Same color, but one connected, one not."""
        grid = np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
        ])
        # Everything is object (non-zero)
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'connected_regions',
            'difficulty': 'medium',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'All colored cells are objects (regardless of connectivity)',
            'success_iou': 1.0
        }
    
    def _puzzle_L_shape_connectivity(self):
        """L-shaped object."""
        grid = np.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'L_shape',
            'difficulty': 'medium',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect L-shaped connected object',
            'success_iou': 1.0
        }
    
    def _puzzle_diagonal_not_connected(self):
        """Diagonally adjacent cells (NOT connected in 4-connectivity)."""
        grid = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        # All are objects, but connectivity not tested (that's topology)
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'diagonal_cells',
            'difficulty': 'medium',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect all colored cells as objects',
            'success_iou': 1.0
        }
    
    def _puzzle_multiple_same_color(self):
        """Multiple regions of same color."""
        grid = np.array([
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'same_color_regions',
            'difficulty': 'medium',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect all same-color regions',
            'success_iou': 1.0
        }
    
    # ===== HARD LEVEL =====
    
    def _puzzle_complex_connected_shape(self):
        """Complex connected shape."""
        grid = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'plus_shape',
            'difficulty': 'hard',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect plus/cross shape',
            'success_iou': 1.0
        }
    
    def _puzzle_touching_different_colors(self):
        """Different colored objects touching."""
        grid = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'touching_colors',
            'difficulty': 'hard',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect all objects (even when touching)',
            'success_iou': 1.0
        }
    
    def _puzzle_nested_different_colors(self):
        """One color inside another."""
        grid = np.array([
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 1, 1, 1],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'nested_colors',
            'difficulty': 'hard',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect nested colored regions',
            'success_iou': 1.0
        }
    
    def _puzzle_partial_overlap(self):
        """Regions that share boundary."""
        grid = np.array([
            [1, 1, 1, 0],
            [1, 2, 1, 0],
            [1, 1, 1, 0],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'shared_boundary',
            'difficulty': 'hard',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect objects with shared boundaries',
            'success_iou': 1.0
        }
    
    # ===== ARC LEVEL =====
    
    def _puzzle_thin_connection(self):
        """Connected by single cell."""
        grid = np.array([
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'thin_connection',
            'difficulty': 'arc',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect thinly connected regions',
            'success_iou': 0.95  # Allow small errors
        }
    
    def _puzzle_surrounding_frame(self):
        """Frame around center."""
        grid = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'frame',
            'difficulty': 'arc',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect frame/border as object',
            'success_iou': 0.95
        }
    
    def _puzzle_diagonal_proximity(self):
        """Diagonally close but not connected."""
        grid = np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'checkerboard',
            'difficulty': 'arc',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect checkerboard pattern',
            'success_iou': 0.95
        }
    
    def _puzzle_color_boundaries(self):
        """Multiple colors forming pattern."""
        grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        expected_mask = (grid != 0).astype(float)
        
        return {
            'name': 'multi_color_grid',
            'difficulty': 'arc',
            'grid': grid,
            'expected_mask': expected_mask,
            'description': 'Detect all colored cells',
            'success_iou': 0.90  # Challenging
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
                'expected_mask': puzzle['expected_mask'].tolist(),
                'description': puzzle['description'],
                'success_iou': puzzle['success_iou']
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        return self.puzzles[idx]


if __name__ == "__main__":
    benchmark = PureObjectCognitionBenchmark()
    
    print("="*70)
    print("PURE OBJECT COGNITION BENCHMARK")
    print("="*70)
    print(f"\nTotal puzzles: {len(benchmark)}")
    print(f"\nBy difficulty:")
    for diff in ['easy', 'medium', 'hard', 'arc']:
        count = len(benchmark.get_by_difficulty(diff))
        print(f"  {diff.upper():8s}: {count} puzzles")
    
    # Save
    filepath = "data/pure_object_cognition_benchmark.json"
    benchmark.save_to_json(filepath)
    print(f"\n✓ Saved to {filepath}")
    
    # Show samples
    print(f"\n{'='*70}")
    print("TASK: Predict object mask (IoU score)")
    print(f"{'='*70}\n")
    
    for diff in ['easy', 'medium', 'hard', 'arc']:
        puzzles = benchmark.get_by_difficulty(diff)
        if puzzles:
            puzzle = puzzles[0]
            print(f"\n📋 {diff.upper()}: {puzzle['name']}")
            print(f"   {puzzle['description']}")
            print(f"   Success IoU: ≥{puzzle['success_iou']:.0%}")
            print(f"   Grid:")
            for row in puzzle['grid']:
                print(f"     {' '.join(str(x) for x in row)}")
            print(f"   Expected mask:")
            for row in puzzle['expected_mask']:
                print(f"     {' '.join('█' if x > 0 else '·' for x in row)}")
    
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA")
    print(f"{'='*70}\n")
    print("Evaluation metric: IoU (Intersection over Union)")
    print("  EASY level:   IoU ≥ 100% (perfect segmentation)")
    print("  MEDIUM level: IoU ≥ 100% (exact boundaries)")
    print("  HARD level:   IoU ≥ 100% (complex shapes)")
    print("  ARC level:    IoU ≥  90% (allow minor errors)")
    print("\nOverall: Average IoU ≥ 95% → STRONG PRIMITIVE ✓")
    print("\nThis tests ONLY object detection/segmentation")
    print("NO counting, NO other primitives needed!")
