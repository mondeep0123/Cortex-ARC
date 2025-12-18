"""Tests for the Color Encoder module."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain.visual.color_encoder import (
    ColorEncoder,
    ARC_COLORS,
    COLOR_PROPERTIES,
    verify_color_encoder
)


class TestColorProperties:
    """Test deterministic color properties (must be 100% accurate)."""
    
    def test_all_colors_have_properties(self):
        """All 10 colors should have properties."""
        assert len(COLOR_PROPERTIES) == 10
        for i in range(10):
            assert i in COLOR_PROPERTIES
    
    def test_color_names_correct(self):
        """Color names should match ARC specification."""
        expected_names = [
            "black", "blue", "red", "green", "yellow",
            "grey", "magenta", "orange", "cyan", "maroon"
        ]
        for i, name in enumerate(expected_names):
            assert COLOR_PROPERTIES[i].name == name
    
    def test_brightness_range(self):
        """Brightness should be between 0 and 1."""
        for i in range(10):
            brightness = COLOR_PROPERTIES[i].brightness
            assert 0.0 <= brightness <= 1.0
    
    def test_black_is_dark(self):
        """Black (0) should be classified as dark."""
        assert COLOR_PROPERTIES[0].is_dark == True
        assert COLOR_PROPERTIES[0].brightness < 0.1
    
    def test_yellow_is_bright(self):
        """Yellow (4) should be bright (not dark)."""
        assert COLOR_PROPERTIES[4].is_dark == False
    
    def test_black_is_background(self):
        """Black (0) should have typical_role = background."""
        assert COLOR_PROPERTIES[0].typical_role == "background"


class TestColorEncoderOneHot:
    """Test one-hot encoding (must be 100% accurate)."""
    
    def test_onehot_shape(self):
        """One-hot should produce correct shape."""
        encoder = ColorEncoder()
        colors = torch.tensor([0, 1, 2])
        onehot = encoder.encode_onehot(colors)
        assert onehot.shape == (3, 10)
    
    def test_onehot_100_percent_accuracy(self):
        """One-hot encoding must be 100% accurate."""
        encoder = ColorEncoder()
        
        for i in range(10):
            colors = torch.tensor([i])
            onehot = encoder.encode_onehot(colors)
            predicted = onehot[0].argmax().item()
            assert predicted == i, f"Color {i} should encode to itself"
    
    def test_onehot_grid(self):
        """One-hot should work on grids."""
        encoder = ColorEncoder()
        grid = torch.tensor([[0, 1], [2, 3]])
        onehot = encoder.encode_onehot(grid)
        assert onehot.shape == (2, 2, 10)
        
        # Verify specific values
        assert onehot[0, 0, 0] == 1.0  # (0,0) is color 0
        assert onehot[0, 1, 1] == 1.0  # (0,1) is color 1
        assert onehot[1, 0, 2] == 1.0  # (1,0) is color 2
        assert onehot[1, 1, 3] == 1.0  # (1,1) is color 3


class TestColorEncoderProperties:
    """Test property encoding (must be 100% accurate)."""
    
    def test_properties_shape(self):
        """Property encoding should produce correct shape."""
        encoder = ColorEncoder()
        colors = torch.tensor([0, 1, 2])
        props = encoder.encode_properties(colors)
        assert props.shape[0] == 3
        assert props.shape[1] == 9  # Number of property features
    
    def test_properties_deterministic(self):
        """Same input should always produce same output."""
        encoder = ColorEncoder()
        colors = torch.tensor([5])
        
        props1 = encoder.encode_properties(colors)
        props2 = encoder.encode_properties(colors)
        
        assert torch.equal(props1, props2)
    
    def test_properties_different_colors(self):
        """Different colors should have different properties."""
        encoder = ColorEncoder()
        
        black_props = encoder.encode_properties(torch.tensor([0]))
        yellow_props = encoder.encode_properties(torch.tensor([4]))
        
        # They should be different
        assert not torch.equal(black_props, yellow_props)


class TestColorEncoderEmbedding:
    """Test learnable embeddings."""
    
    def test_embedding_shape(self):
        """Embedding should produce correct shape."""
        encoder = ColorEncoder(embedding_dim=32)
        colors = torch.tensor([0, 1, 2])
        emb = encoder.encode_embedding(colors)
        assert emb.shape == (3, 32)
    
    def test_embedding_learnable(self):
        """Embeddings should have gradients."""
        encoder = ColorEncoder()
        colors = torch.tensor([0])
        emb = encoder.encode_embedding(colors)
        
        # Embeddings should require grad
        assert emb.requires_grad


class TestColorEncoderFull:
    """Test full encoding pipeline."""
    
    def test_forward_shape(self):
        """Forward pass should produce correct shape."""
        encoder = ColorEncoder(embedding_dim=32)
        colors = torch.tensor([[0, 1], [2, 3]])
        output = encoder(colors)
        
        # output_dim = embedding_dim + 10 (onehot) + 9 (properties)
        expected_dim = 32 + 10 + 9
        assert output.shape == (2, 2, expected_dim)
    
    def test_output_dim_attribute(self):
        """output_dim attribute should match actual output."""
        encoder = ColorEncoder(embedding_dim=32)
        colors = torch.tensor([0])
        output = encoder(colors)
        
        assert output.shape[-1] == encoder.output_dim
    
    def test_encode_grid(self):
        """encode_grid should work with numpy arrays."""
        encoder = ColorEncoder()
        grid = np.array([[0, 1, 2], [3, 4, 5]])
        output = encoder.encode_grid(grid)
        
        assert output.shape == (2, 3, encoder.output_dim)


class TestColorSimilarity:
    """Test color similarity functions."""
    
    def test_color_distance_self(self):
        """Distance to self should be 0."""
        encoder = ColorEncoder()
        for i in range(10):
            dist = encoder.color_distance(i, i)
            assert dist == 0.0
    
    def test_color_distance_symmetric(self):
        """Distance should be symmetric."""
        encoder = ColorEncoder()
        for i in range(10):
            for j in range(10):
                dist_ij = encoder.color_distance(i, j)
                dist_ji = encoder.color_distance(j, i)
                assert abs(dist_ij - dist_ji) < 1e-6
    
    def test_most_similar_excludes_self(self):
        """most_similar_colors should not include the query color."""
        encoder = ColorEncoder()
        similar = encoder.most_similar_colors(0, top_k=3)
        
        for color_id, _ in similar:
            assert color_id != 0


class TestColorEncoderAccuracy:
    """Test that deterministic components achieve 100% accuracy."""
    
    def test_100_percent_accuracy_onehot(self):
        """One-hot encoding must achieve 100% accuracy."""
        encoder = ColorEncoder()
        
        # Test all colors
        all_colors = torch.arange(10)
        onehot = encoder.encode_onehot(all_colors)
        
        # Decode by taking argmax
        decoded = onehot.argmax(dim=-1)
        
        # ALL must match
        accuracy = (decoded == all_colors).float().mean().item()
        assert accuracy == 1.0, f"One-hot accuracy is {accuracy*100}%, must be 100%"
    
    def test_100_percent_accuracy_reconstruction(self):
        """Color should be perfectly reconstructable from one-hot."""
        encoder = ColorEncoder()
        
        # Create random grid
        np.random.seed(42)
        grid = np.random.randint(0, 10, size=(10, 10))
        
        # Encode
        colors = torch.tensor(grid)
        onehot = encoder.encode_onehot(colors)
        
        # Reconstruct
        reconstructed = onehot.argmax(dim=-1).numpy()
        
        # Must be PERFECT
        assert np.array_equal(grid, reconstructed)


if __name__ == "__main__":
    # Run verification
    verify_color_encoder()
    
    # Run tests
    pytest.main([__file__, "-v"])
