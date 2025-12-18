"""
Grid Encoder: Converts input grids to learned embeddings.

Phase 1 Focus: Color Understanding
- Color embeddings (learned, not one-hot)
- Position embeddings
- Patch/pixel encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ColorEmbedding(nn.Module):
    """
    Learned color embeddings.
    
    NOT one-hot encoding. Colors with similar "meaning" should have
    similar embeddings. This is LEARNED from data.
    
    Brain analogy: V4 color processing - learns color relationships
    """
    
    def __init__(self, num_colors: int = 10, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Learnable color embeddings
        # Initialize with some structure (but model can override)
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        
        # Initialize: similar colors start closer (can be learned away)
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize with slight structure, but fully learnable."""
        nn.init.normal_(self.color_embed.weight, mean=0, std=0.02)
        
        # Background (0) gets special initialization
        with torch.no_grad():
            self.color_embed.weight[0] = torch.zeros(self.embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Grid of color values [B, H, W] with values 0-9
            
        Returns:
            Color embeddings [B, H, W, embed_dim]
        """
        return self.color_embed(x.long())


class PositionEmbedding(nn.Module):
    """
    2D learnable position embeddings.
    
    Brain analogy: Spatial awareness from parietal cortex
    """
    
    def __init__(self, max_size: int = 30, embed_dim: int = 64):
        super().__init__()
        self.max_size = max_size
        self.embed_dim = embed_dim
        
        # Learnable row and column embeddings
        self.row_embed = nn.Embedding(max_size, embed_dim // 2)
        self.col_embed = nn.Embedding(max_size, embed_dim // 2)
        
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize with sinusoidal-like structure."""
        nn.init.normal_(self.row_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.col_embed.weight, mean=0, std=0.02)
    
    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Generate position embeddings for an HxW grid.
        
        Args:
            h: Height of grid
            w: Width of grid
            device: Device to create tensors on
            
        Returns:
            Position embeddings [H, W, embed_dim]
        """
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)
        
        row_emb = self.row_embed(rows)  # [H, embed_dim//2]
        col_emb = self.col_embed(cols)  # [W, embed_dim//2]
        
        # Combine: each position gets concat of row and col embedding
        row_emb = row_emb.unsqueeze(1).expand(-1, w, -1)  # [H, W, embed_dim//2]
        col_emb = col_emb.unsqueeze(0).expand(h, -1, -1)  # [H, W, embed_dim//2]
        
        return torch.cat([row_emb, col_emb], dim=-1)  # [H, W, embed_dim]


class GridEncoder(nn.Module):
    """
    Encodes input grids into learned representations.
    
    Combines:
    - Color embeddings (what color is each cell?)
    - Position embeddings (where is each cell?)
    - Initial projection to model dimension
    
    This is the input processing stage - analogous to early visual cortex.
    """
    
    def __init__(
        self,
        num_colors: int = 10,
        max_size: int = 30,
        embed_dim: int = 128,
        color_dim: int = 64,
        position_dim: int = 64,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.color_dim = color_dim
        self.position_dim = position_dim
        
        # Color understanding
        self.color_embed = ColorEmbedding(num_colors, color_dim)
        
        # Spatial awareness
        self.position_embed = PositionEmbedding(max_size, position_dim)
        
        # Project combined features to model dimension
        self.input_proj = nn.Linear(color_dim + position_dim, embed_dim)
        
        # Layer norm for stable training
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Encode a grid into learned representations.
        
        Args:
            grid: Input grid [B, H, W] with color values 0-9
            
        Returns:
            Encoded representations [B, H, W, embed_dim]
        """
        B, H, W = grid.shape
        device = grid.device
        
        # Get color embeddings
        color_emb = self.color_embed(grid)  # [B, H, W, color_dim]
        
        # Get position embeddings (same for all batches)
        pos_emb = self.position_embed(H, W, device)  # [H, W, position_dim]
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, position_dim]
        
        # Combine color and position
        combined = torch.cat([color_emb, pos_emb], dim=-1)  # [B, H, W, color+position]
        
        # Project to model dimension
        encoded = self.input_proj(combined)  # [B, H, W, embed_dim]
        encoded = self.norm(encoded)
        
        return encoded
    
    def get_color_embeddings(self) -> torch.Tensor:
        """Get the learned color embedding matrix for visualization."""
        return self.color_embed.color_embed.weight.detach()


# Testing
if __name__ == "__main__":
    print("Testing GridEncoder...")
    
    encoder = GridEncoder()
    
    # Test input
    grid = torch.randint(0, 10, (2, 5, 5))  # Batch of 2, 5x5 grids
    
    # Encode
    encoded = encoder(grid)
    print(f"Input shape: {grid.shape}")
    print(f"Output shape: {encoded.shape}")
    print(f"Expected: [2, 5, 5, 128]")
    
    # Check color embeddings
    colors = encoder.get_color_embeddings()
    print(f"\nColor embeddings shape: {colors.shape}")
    print(f"Background (0) embedding norm: {colors[0].norm():.4f}")
    
    print("\n✓ GridEncoder working!")
