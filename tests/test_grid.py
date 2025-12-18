"""Tests for Grid class."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.grid import Grid


class TestGridCreation:
    """Test grid creation methods."""
    
    def test_from_list(self):
        """Test creating grid from nested list."""
        data = [[0, 1, 2], [3, 4, 5]]
        grid = Grid.from_list(data)
        
        assert grid.height == 2
        assert grid.width == 3
        assert grid[0, 0] == 0
        assert grid[1, 2] == 5
    
    def test_from_array(self):
        """Test creating grid from numpy array."""
        data = np.array([[1, 2], [3, 4]], dtype=np.int8)
        grid = Grid(data=data)
        
        assert grid.height == 2
        assert grid.width == 2
    
    def test_zeros(self):
        """Test creating zero-filled grid."""
        grid = Grid.zeros(5, 10)
        
        assert grid.height == 5
        assert grid.width == 10
        assert np.all(grid.data == 0)
    
    def test_ones(self):
        """Test creating grid filled with color."""
        grid = Grid.ones(3, 3, color=5)
        
        assert grid.shape == (3, 3)
        assert np.all(grid.data == 5)
    
    def test_invalid_color_raises(self):
        """Test that invalid colors raise assertion."""
        with pytest.raises(AssertionError):
            Grid(data=np.array([[10]]))  # Color > 9
        
        with pytest.raises(AssertionError):
            Grid(data=np.array([[-1]]))  # Color < 0
    
    def test_invalid_size_raises(self):
        """Test that invalid sizes raise assertion."""
        with pytest.raises(AssertionError):
            Grid(data=np.zeros((0, 5), dtype=np.int8))  # Height 0
        
        with pytest.raises(AssertionError):
            Grid(data=np.zeros((31, 5), dtype=np.int8))  # Height > 30


class TestGridProperties:
    """Test grid property methods."""
    
    def test_unique_colors(self):
        """Test finding unique colors."""
        grid = Grid.from_list([[0, 1, 2], [0, 1, 2]])
        assert grid.unique_colors() == [0, 1, 2]
    
    def test_color_counts(self):
        """Test counting colors."""
        grid = Grid.from_list([[0, 0, 1], [1, 1, 1]])
        counts = grid.color_counts()
        
        assert counts[0] == 2
        assert counts[1] == 4
    
    def test_background_color(self):
        """Test detecting background color."""
        grid = Grid.from_list([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert grid.background_color() == 0
        
        grid2 = Grid.from_list([[5, 5, 5], [5, 1, 5], [5, 5, 5]])
        assert grid2.background_color() == 5


class TestGridTransforms:
    """Test grid transformations."""
    
    def test_rotate_90(self):
        """Test 90-degree clockwise rotation."""
        grid = Grid.from_list([[1, 2], [3, 4]])
        rotated = grid.rotate_90()
        
        expected = [[3, 1], [4, 2]]
        assert rotated.to_list() == expected
    
    def test_rotate_180(self):
        """Test 180-degree rotation."""
        grid = Grid.from_list([[1, 2], [3, 4]])
        rotated = grid.rotate_180()
        
        expected = [[4, 3], [2, 1]]
        assert rotated.to_list() == expected
    
    def test_flip_horizontal(self):
        """Test horizontal flip."""
        grid = Grid.from_list([[1, 2, 3], [4, 5, 6]])
        flipped = grid.flip_horizontal()
        
        expected = [[3, 2, 1], [6, 5, 4]]
        assert flipped.to_list() == expected
    
    def test_flip_vertical(self):
        """Test vertical flip."""
        grid = Grid.from_list([[1, 2], [3, 4]])
        flipped = grid.flip_vertical()
        
        expected = [[3, 4], [1, 2]]
        assert flipped.to_list() == expected
    
    def test_transpose(self):
        """Test transpose."""
        grid = Grid.from_list([[1, 2, 3], [4, 5, 6]])
        transposed = grid.transpose()
        
        expected = [[1, 4], [2, 5], [3, 6]]
        assert transposed.to_list() == expected
    
    def test_scale(self):
        """Test scaling."""
        grid = Grid.from_list([[1, 2], [3, 4]])
        scaled = grid.scale(2)
        
        assert scaled.height == 4
        assert scaled.width == 4
        assert scaled[0, 0] == 1
        assert scaled[0, 1] == 1
        assert scaled[1, 0] == 1
    
    def test_replace_color(self):
        """Test color replacement."""
        grid = Grid.from_list([[0, 1, 0], [1, 0, 1]])
        replaced = grid.replace_color(1, 5)
        
        expected = [[0, 5, 0], [5, 0, 5]]
        assert replaced.to_list() == expected


class TestGridComparison:
    """Test grid comparison methods."""
    
    def test_equality(self):
        """Test grid equality."""
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 2], [3, 4]])
        g3 = Grid.from_list([[1, 2], [3, 5]])
        
        assert g1 == g2
        assert g1 != g3
    
    def test_similarity(self):
        """Test similarity calculation."""
        g1 = Grid.from_list([[1, 1], [1, 1]])
        g2 = Grid.from_list([[1, 1], [1, 0]])
        
        assert g1.similarity(g1) == 1.0
        assert g1.similarity(g2) == 0.75
    
    def test_hash(self):
        """Test grid hashing for dict/set usage."""
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 2], [3, 4]])
        
        assert hash(g1) == hash(g2)
        
        # Can be used in set
        grid_set = {g1, g2}
        assert len(grid_set) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
