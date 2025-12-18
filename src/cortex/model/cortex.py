"""
Cortex Model: Unified Reasoning Core

A single neural network that learns all cognitive abilities through
shared weights. Abilities emerge from training, not from separate modules.

Architecture:
    Input → Encoder → Reasoning Core → Decoder → Output
    
The Reasoning Core uses attention to learn:
    - Color relationships (which colors relate?)
    - Spatial relationships (which positions relate?)
    - Pattern recognition (what repeats?)
    - Object detection (what groups together?)
    - Transformation detection (how does input map to output?)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .encoder import GridEncoder
from .decoder import GridDecoder


class ReasoningBlock(nn.Module):
    """
    A single reasoning step.
    
    Uses self-attention to relate all positions to each other,
    learning which positions are relevant to which.
    
    Brain analogy: A layer of cortical processing
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Self-attention: learn which positions relate
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # MLP: process attended features
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one reasoning step.
        
        Args:
            x: Input [B, N, embed_dim] where N = H*W (flattened grid)
            
        Returns:
            Processed output [B, N, embed_dim]
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class ReasoningCore(nn.Module):
    """
    The reasoning core: multiple reasoning blocks.
    
    This is where cognitive abilities emerge from training.
    No hardcoded color/spatial/pattern modules - all learned.
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            ReasoningBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """
        Apply reasoning.
        
        Args:
            x: Encoded grid [B, H, W, embed_dim]
            shape: Original (H, W) for reshaping
            
        Returns:
            Reasoned output [B, H, W, embed_dim]
        """
        B, H, W, D = x.shape
        
        # Flatten spatial dimensions for attention
        x = x.view(B, H * W, D)  # [B, N, D]
        
        # Apply reasoning layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Reshape back to grid
        x = x.view(B, H, W, D)
        
        return x


class CortexModel(nn.Module):
    """
    Cortex: Unified Reasoning Model
    
    One model that learns all cognitive abilities through shared weights.
    
    Training proceeds in phases:
        Phase 1: Color tasks (learn color understanding)
        Phase 2: Spatial tasks (learn spatial reasoning)
        Phase 3: Pattern tasks (learn pattern recognition)
        Phase 4: Full ARC tasks (combine all abilities)
    
    But it's always the SAME model - abilities emerge, not separate modules.
    """
    
    def __init__(
        self,
        num_colors: int = 10,
        max_size: int = 30,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        
        # Encoder: grid → embeddings
        self.encoder = GridEncoder(
            num_colors=num_colors,
            max_size=max_size,
            embed_dim=embed_dim,
            color_dim=embed_dim // 2,
            position_dim=embed_dim // 2,
        )
        
        # Reasoning core: where abilities emerge
        self.reasoning = ReasoningCore(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Decoder: embeddings → grid
        self.decoder = GridDecoder(
            embed_dim=embed_dim,
            num_colors=num_colors,
        )
    
    def forward(
        self,
        input_grid: torch.Tensor,
        target_grid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_grid: Input grid [B, H, W]
            target_grid: Target grid [B, H, W] (for training loss)
            
        Returns:
            (output_logits, loss) where loss is None if no target
        """
        B, H, W = input_grid.shape
        
        # Encode
        encoded = self.encoder(input_grid)  # [B, H, W, D]
        
        # Reason
        reasoned = self.reasoning(encoded, (H, W))  # [B, H, W, D]
        
        # Decode
        logits = self.decoder(reasoned)  # [B, H, W, num_colors]
        
        # Compute loss if target provided
        loss = None
        if target_grid is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_colors),
                target_grid.view(-1).long(),
            )
        
        return logits, loss
    
    def predict(self, input_grid: torch.Tensor) -> torch.Tensor:
        """
        Predict output grid.
        
        Args:
            input_grid: Input grid [B, H, W]
            
        Returns:
            Predicted grid [B, H, W]
        """
        logits, _ = self.forward(input_grid)
        return logits.argmax(dim=-1)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Testing
if __name__ == "__main__":
    print("Testing CortexModel...")
    
    model = CortexModel(
        num_colors=10,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
    )
    
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    input_grid = torch.randint(0, 10, (2, 5, 5))
    target_grid = torch.randint(0, 10, (2, 5, 5))
    
    logits, loss = model(input_grid, target_grid)
    prediction = model.predict(input_grid)
    
    print(f"\nInput shape: {input_grid.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Prediction shape: {prediction.shape}")
    
    print("\n✓ CortexModel working!")
