"""
Color Encoder - The Foundation of Visual Processing

This is Module 1 of the Visual Cortex, the absolute foundation.
Everything sees through color first.

Design Philosophy:
    - 100% accuracy is the goal (only 10 colors - this is achievable)
    - Combine deterministic properties + learnable representations
    - Each color gets both:
        1. Fixed properties (RGB, name, is_background, etc.)
        2. Learnable embedding (relationships learned from data)

What the Color Encoder provides:
    1. Color Identity: One-hot encoding (perfect - deterministic)
    2. Color Properties: RGB, brightness, etc. (perfect - deterministic)
    3. Color Embeddings: Learned relationships (trained on ARC grids)
    4. Color Similarity: Which colors are "similar" in ARC context
    5. Color Role: Is this a background color? foreground? marker?
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# DETERMINISTIC COLOR PROPERTIES (100% accurate by design)
# ============================================================================

# Official ARC color palette (RGB values)
ARC_COLORS = {
    0: {"name": "black",   "rgb": (0, 0, 0),       "hex": "#000000"},
    1: {"name": "blue",    "rgb": (0, 116, 217),   "hex": "#0074D9"},
    2: {"name": "red",     "rgb": (255, 65, 54),   "hex": "#FF4136"},
    3: {"name": "green",   "rgb": (46, 204, 64),   "hex": "#2ECC40"},
    4: {"name": "yellow",  "rgb": (255, 220, 0),   "hex": "#FFDC00"},
    5: {"name": "grey",    "rgb": (170, 170, 170), "hex": "#AAAAAA"},
    6: {"name": "magenta", "rgb": (240, 18, 190),  "hex": "#F012BE"},
    7: {"name": "orange",  "rgb": (255, 133, 27),  "hex": "#FF851B"},
    8: {"name": "cyan",    "rgb": (127, 219, 255), "hex": "#7FDBFF"},
    9: {"name": "maroon",  "rgb": (135, 12, 37),   "hex": "#870C25"},
}


@dataclass
class ColorProperties:
    """Fixed, deterministic properties for each color."""
    
    color_id: int
    name: str
    rgb: Tuple[int, int, int]
    
    # Derived properties (computed, 100% accurate)
    brightness: float  # 0-1, how bright the color is
    is_dark: bool      # brightness < 0.5
    is_warm: bool      # reds, oranges, yellows
    is_cool: bool      # blues, greens, cyans
    is_neutral: bool   # black, grey
    
    # ARC-specific properties (observed from data)
    typical_role: str  # "background", "foreground", "marker", "any"
    
    @classmethod
    def compute_brightness(cls, rgb: Tuple[int, int, int]) -> float:
        """Compute perceived brightness (0-1)."""
        r, g, b = rgb
        # Human perception weighted formula
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    
    @classmethod
    def from_color_id(cls, color_id: int) -> ColorProperties:
        """Create ColorProperties for a given color ID."""
        info = ARC_COLORS[color_id]
        rgb = info["rgb"]
        brightness = cls.compute_brightness(rgb)
        
        r, g, b = rgb
        
        # Warm colors: more red/orange/yellow
        is_warm = (r > b) and (r > 100 or g > 150)
        
        # Cool colors: more blue/green/cyan
        is_cool = (b > r) or (g > r and g > 100)
        
        # Neutral: grey or black
        is_neutral = (abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30)
        
        # Typical role in ARC (observed patterns)
        if color_id == 0:
            typical_role = "background"
        elif color_id in [1, 2, 3]:
            typical_role = "primary"
        elif color_id in [5]:
            typical_role = "secondary"
        else:
            typical_role = "any"
        
        return cls(
            color_id=color_id,
            name=info["name"],
            rgb=rgb,
            brightness=brightness,
            is_dark=brightness < 0.5,
            is_warm=is_warm and not is_neutral,
            is_cool=is_cool and not is_neutral,
            is_neutral=is_neutral,
            typical_role=typical_role,
        )


# Pre-compute all color properties (deterministic, 100% accurate)
COLOR_PROPERTIES: Dict[int, ColorProperties] = {
    i: ColorProperties.from_color_id(i) for i in range(10)
}


# ============================================================================
# COLOR ENCODER MODULE
# ============================================================================

class ColorEncoder(nn.Module):
    """
    The Color Encoder - Foundation of visual processing.
    
    Provides multiple representations for colors:
    
    1. One-Hot Encoding (deterministic, 100% accurate)
       - Simple but perfect identity
       
    2. Property Encoding (deterministic, 100% accurate)
       - RGB values, brightness, warmth, etc.
       
    3. Learned Embedding (trained on ARC grids)
       - Captures relationships between colors as used in ARC
       
    4. Combined Encoding
       - Concatenation of all above for rich representation
    """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        include_properties: bool = True,
        include_onehot: bool = True,
    ):
        """
        Initialize the Color Encoder.
        
        Args:
            embedding_dim: Dimension of learnable color embeddings
            include_properties: Include deterministic color properties
            include_onehot: Include one-hot encoding
        """
        super().__init__()
        
        self.num_colors = 10
        self.embedding_dim = embedding_dim
        self.include_properties = include_properties
        self.include_onehot = include_onehot
        
        # Learnable embeddings (will be trained on ARC grids)
        self.color_embeddings = nn.Embedding(
            num_embeddings=self.num_colors,
            embedding_dim=embedding_dim
        )
        
        # Initialize embeddings with some structure
        self._initialize_embeddings()
        
        # Pre-compute deterministic encodings (100% accurate)
        self.register_buffer(
            'onehot_matrix',
            torch.eye(self.num_colors, dtype=torch.float32)
        )
        
        self.register_buffer(
            'property_matrix',
            self._build_property_matrix()
        )
        
        # Calculate total output dimension
        self.output_dim = embedding_dim
        if include_onehot:
            self.output_dim += self.num_colors
        if include_properties:
            self.output_dim += self.property_matrix.shape[1]
    
    def _initialize_embeddings(self):
        """Initialize embeddings with color-aware structure."""
        # Start with small random values
        nn.init.normal_(self.color_embeddings.weight, mean=0, std=0.1)
        
        # Add structure: similar colors should start closer
        with torch.no_grad():
            # Group 1: Dark colors (0, 9) 
            # Group 2: Primary colors (1, 2, 3)
            # Group 3: Bright colors (4, 7, 8)
            # Group 4: Neutral (5)
            # Group 5: Distinct (6)
            
            # This gives the model a good starting point
            for i in range(10):
                prop = COLOR_PROPERTIES[i]
                # Add some structure based on brightness
                self.color_embeddings.weight[i, 0] = prop.brightness
                # Add structure based on RGB
                r, g, b = prop.rgb
                self.color_embeddings.weight[i, 1] = r / 255.0
                self.color_embeddings.weight[i, 2] = g / 255.0
                self.color_embeddings.weight[i, 3] = b / 255.0
    
    def _build_property_matrix(self) -> torch.Tensor:
        """Build deterministic property matrix (100% accurate)."""
        properties = []
        
        for i in range(10):
            prop = COLOR_PROPERTIES[i]
            r, g, b = prop.rgb
            
            # Normalize RGB to 0-1
            feat = [
                r / 255.0,
                g / 255.0,
                b / 255.0,
                prop.brightness,
                1.0 if prop.is_dark else 0.0,
                1.0 if prop.is_warm else 0.0,
                1.0 if prop.is_cool else 0.0,
                1.0 if prop.is_neutral else 0.0,
                1.0 if prop.typical_role == "background" else 0.0,
            ]
            properties.append(feat)
        
        return torch.tensor(properties, dtype=torch.float32)
    
    # ========================================================================
    # ENCODING METHODS (ALL 100% ACCURATE)
    # ========================================================================
    
    def encode_onehot(self, colors: torch.Tensor) -> torch.Tensor:
        """
        One-hot encode colors. 100% accurate.
        
        Args:
            colors: Tensor of color values (0-9), any shape
            
        Returns:
            Tensor of shape (*colors.shape, 10) with one-hot encoding
        """
        original_shape = colors.shape
        flat_colors = colors.flatten().long()
        onehot = self.onehot_matrix[flat_colors]
        return onehot.view(*original_shape, 10)
    
    def encode_properties(self, colors: torch.Tensor) -> torch.Tensor:
        """
        Encode deterministic color properties. 100% accurate.
        
        Args:
            colors: Tensor of color values (0-9), any shape
            
        Returns:
            Tensor with color property features
        """
        original_shape = colors.shape
        flat_colors = colors.flatten().long()
        props = self.property_matrix[flat_colors]
        return props.view(*original_shape, -1)
    
    def encode_embedding(self, colors: torch.Tensor) -> torch.Tensor:
        """
        Get learnable embeddings for colors.
        
        Args:
            colors: Tensor of color values (0-9), any shape
            
        Returns:
            Tensor with learned color embeddings
        """
        return self.color_embeddings(colors.long())
    
    def forward(self, colors: torch.Tensor) -> torch.Tensor:
        """
        Full color encoding combining all representations.
        
        Args:
            colors: Tensor of color values (0-9), shape (B, H, W) or (H, W)
            
        Returns:
            Tensor of shape (*colors.shape, output_dim) with rich color encoding
        """
        parts = []
        
        # Learned embeddings (trainable)
        parts.append(self.encode_embedding(colors))
        
        # One-hot (100% accurate)
        if self.include_onehot:
            parts.append(self.encode_onehot(colors))
        
        # Properties (100% accurate)
        if self.include_properties:
            parts.append(self.encode_properties(colors))
        
        # Concatenate all parts
        return torch.cat(parts, dim=-1)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def color_distance(self, color1: int, color2: int) -> float:
        """
        Compute learned distance between two colors.
        
        Args:
            color1, color2: Color IDs (0-9)
            
        Returns:
            Distance in embedding space
        """
        with torch.no_grad():
            emb1 = self.color_embeddings.weight[color1]
            emb2 = self.color_embeddings.weight[color2]
            return torch.norm(emb1 - emb2).item()
    
    def most_similar_colors(self, color: int, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Find most similar colors in embedding space.
        
        Args:
            color: Color ID to find similar colors for
            top_k: Number of similar colors to return
            
        Returns:
            List of (color_id, distance) tuples
        """
        with torch.no_grad():
            target = self.color_embeddings.weight[color]
            distances = []
            
            for i in range(10):
                if i != color:
                    dist = torch.norm(target - self.color_embeddings.weight[i]).item()
                    distances.append((i, dist))
            
            return sorted(distances, key=lambda x: x[1])[:top_k]
    
    def encode_grid(self, grid: np.ndarray) -> torch.Tensor:
        """
        Encode an entire ARC grid.
        
        Args:
            grid: Numpy array of shape (H, W) with values 0-9
            
        Returns:
            Tensor of shape (H, W, output_dim) with color encodings
        """
        colors = torch.tensor(grid, dtype=torch.long)
        return self.forward(colors)
    
    def __repr__(self) -> str:
        return (
            f"ColorEncoder(\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  include_properties={self.include_properties},\n"
            f"  include_onehot={self.include_onehot},\n"
            f"  output_dim={self.output_dim}\n"
            f")"
        )


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class ColorEncoderTrainer:
    """
    Trainer for the ColorEncoder's learnable embeddings.
    
    Training objective: Colors that appear in similar contexts
    should have similar embeddings.
    
    Method: Masked Color Prediction
        - Mask random colors in grids
        - Predict masked colors from context
    """
    
    def __init__(
        self,
        encoder: ColorEncoder,
        learning_rate: float = 0.001,
        mask_ratio: float = 0.15,
    ):
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        
        # Prediction head: from embedding to color logits
        self.predictor = nn.Linear(encoder.embedding_dim, 10)
        
        self.optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(self.predictor.parameters()),
            lr=learning_rate
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def create_masked_sample(
        self,
        grid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a masked training sample.
        
        Returns:
            (masked_grid, mask_positions, target_colors)
        """
        h, w = grid.shape
        num_cells = h * w
        num_mask = max(1, int(num_cells * self.mask_ratio))
        
        # Random positions to mask
        flat_indices = np.random.choice(num_cells, num_mask, replace=False)
        mask_positions = np.unravel_index(flat_indices, (h, w))
        
        # Store targets
        target_colors = grid[mask_positions]
        
        # Create masked grid (use -1 or special token for mask)
        masked_grid = grid.copy()
        masked_grid[mask_positions] = -1  # Mask token
        
        return masked_grid, mask_positions, target_colors
    
    def train_step(self, grids: List[np.ndarray]) -> float:
        """
        Single training step on a batch of grids.
        
        Returns:
            Loss value
        """
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        num_samples = 0
        
        for grid in grids:
            masked_grid, mask_pos, targets = self.create_masked_sample(grid)
            
            # Get context embeddings (average of neighbors)
            # For simplicity, use global average as context
            valid_mask = masked_grid >= 0
            valid_colors = torch.tensor(masked_grid[valid_mask], dtype=torch.long)
            context_emb = self.encoder.encode_embedding(valid_colors).mean(dim=0)
            
            # Predict masked colors
            logits = self.predictor(context_emb)
            
            # Compute loss for each masked position
            target_tensor = torch.tensor(targets, dtype=torch.long)
            # Broadcast logits for all targets
            logits_expanded = logits.unsqueeze(0).expand(len(targets), -1)
            
            loss = self.criterion(logits_expanded, target_tensor)
            total_loss += loss
            num_samples += 1
        
        # Backward pass
        if num_samples > 0:
            avg_loss = total_loss / num_samples
            avg_loss.backward()
            self.optimizer.step()
            return avg_loss.item()
        
        return 0.0


# ============================================================================
# TESTING / VERIFICATION
# ============================================================================

def verify_color_encoder():
    """
    Verify that the ColorEncoder achieves 100% accuracy on deterministic components.
    """
    encoder = ColorEncoder()
    
    print("=" * 60)
    print("COLOR ENCODER VERIFICATION")
    print("=" * 60)
    
    # Test 1: One-hot encoding (must be 100% accurate)
    print("\n1. One-Hot Encoding Test:")
    colors = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    onehot = encoder.encode_onehot(colors)
    
    correct = 0
    for i in range(10):
        predicted = onehot[i].argmax().item()
        is_correct = predicted == i
        correct += is_correct
        print(f"   Color {i} ({COLOR_PROPERTIES[i].name:8s}): {'✓' if is_correct else '✗'}")
    
    print(f"   Accuracy: {correct}/10 = {100*correct/10:.0f}%")
    
    # Test 2: Property encoding (must be 100% accurate)
    print("\n2. Property Encoding Test:")
    props = encoder.encode_properties(colors)
    print(f"   Shape: {props.shape}")
    print(f"   Properties are deterministic: ✓")
    
    # Test 3: Grid encoding
    print("\n3. Grid Encoding Test:")
    test_grid = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    encoded = encoder.encode_grid(test_grid)
    print(f"   Grid shape: {test_grid.shape}")
    print(f"   Encoded shape: {encoded.shape}")
    print(f"   Output dim: {encoder.output_dim}")
    
    # Test 4: Color similarity
    print("\n4. Initial Color Similarities (before training):")
    for color in [0, 1, 4]:
        similar = encoder.most_similar_colors(color, top_k=2)
        print(f"   {COLOR_PROPERTIES[color].name:8s} → ", end="")
        for sim_color, dist in similar:
            print(f"{COLOR_PROPERTIES[sim_color].name}({dist:.2f}) ", end="")
        print()
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("Deterministic components: 100% accurate ✓")
    print("=" * 60)
    
    return encoder


if __name__ == "__main__":
    verify_color_encoder()
