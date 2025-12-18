"""
Few-Shot ARC Model

Architecture that actually learns from examples:

1. Encode training examples (input → output pairs)
2. Extract transformation pattern
3. Apply to test input

This is how real ARC solving works:
  - See examples → Learn rule → Apply to test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .encoder import GridEncoder
from .decoder import GridDecoder


class ExampleEncoder(nn.Module):
    """
    Encodes a single (input, output) example pair.
    
    Learns what transformation was applied.
    """
    
    def __init__(self, embed_dim: int = 128, max_size: int = 30):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Encode input and output separately
        self.grid_encoder = GridEncoder(
            num_colors=10,
            max_size=max_size,
            embed_dim=embed_dim,
        )
        
        # Combine input and output representations
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        # Pool over spatial dimensions to get single vector
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(
        self, 
        input_grid: torch.Tensor,  # [B, H1, W1]
        output_grid: torch.Tensor  # [B, H2, W2]
    ) -> torch.Tensor:
        """
        Encode an input-output pair.
        
        Returns:
            Pattern embedding [B, embed_dim]
        """
        # Encode both grids
        inp_emb = self.grid_encoder(input_grid)  # [B, H1, W1, D]
        out_emb = self.grid_encoder(output_grid)  # [B, H2, W2, D]
        
        # Pool to single vector each
        B = input_grid.shape[0]
        inp_pooled = inp_emb.view(B, -1, self.embed_dim).mean(dim=1)  # [B, D]
        out_pooled = out_emb.view(B, -1, self.embed_dim).mean(dim=1)  # [B, D]
        
        # Combine to get transformation pattern
        combined = torch.cat([inp_pooled, out_pooled], dim=-1)  # [B, 2D]
        pattern = self.combine(combined)  # [B, D]
        
        return pattern


class PatternAggregator(nn.Module):
    """
    Aggregates patterns from multiple training examples.
    
    Uses attention to weight examples by importance.
    """
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, patterns: torch.Tensor) -> torch.Tensor:
        """
        Aggregate multiple pattern embeddings.
        
        Args:
            patterns: [B, N, D] where N is number of examples
            
        Returns:
            Aggregated pattern [B, D]
        """
        # Self-attention over examples
        attended, _ = self.attention(patterns, patterns, patterns)
        attended = self.norm(attended)
        
        # Average pool over examples
        aggregated = attended.mean(dim=1)  # [B, D]
        
        return aggregated


class FewShotReasoner(nn.Module):
    """
    Applies learned pattern to test input.
    """
    
    def __init__(self, embed_dim: int = 128, num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Project pattern to condition the transformer
        self.pattern_proj = nn.Linear(embed_dim, embed_dim)
        
        # Transformer layers for reasoning
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        test_emb: torch.Tensor,  # [B, H, W, D]
        pattern: torch.Tensor,   # [B, D]
    ) -> torch.Tensor:
        """
        Apply pattern to test input.
        
        Returns:
            Transformed embeddings [B, H, W, D]
        """
        B, H, W, D = test_emb.shape
        
        # Flatten spatial dimensions
        x = test_emb.view(B, H * W, D)
        
        # Add pattern as conditioning (broadcast and add)
        pattern_cond = self.pattern_proj(pattern).unsqueeze(1)  # [B, 1, D]
        x = x + pattern_cond
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Reshape back to grid
        x = x.view(B, H, W, D)
        
        return x


class FewShotARC(nn.Module):
    """
    Complete Few-Shot ARC Model.
    
    Input: Training examples + Test input
    Output: Predicted test output
    
    Architecture:
        1. Encode each training (input, output) pair → pattern embeddings
        2. Aggregate patterns → single transformation pattern
        3. Encode test input
        4. Apply pattern to test → reasoned embeddings
        5. Decode to output grid
    """
    
    def __init__(
        self,
        num_colors: int = 10,
        max_size: int = 30,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_colors = num_colors
        
        # 1. Encode training examples
        self.example_encoder = ExampleEncoder(embed_dim, max_size)
        
        # 2. Aggregate patterns
        self.pattern_aggregator = PatternAggregator(embed_dim, num_heads)
        
        # 3. Encode test input
        self.test_encoder = GridEncoder(num_colors, max_size, embed_dim)
        
        # 4. Apply pattern
        self.reasoner = FewShotReasoner(embed_dim, num_layers, num_heads)
        
        # 5. Decode output
        self.decoder = GridDecoder(embed_dim, num_colors)
    
    def forward(
        self,
        train_inputs: List[torch.Tensor],   # List of [H, W] grids
        train_outputs: List[torch.Tensor],  # List of [H, W] grids
        test_input: torch.Tensor,           # [H, W] grid
        test_output: Optional[torch.Tensor] = None,  # [H, W] for training
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            train_inputs: List of training input grids
            train_outputs: List of training output grids
            test_input: Test input grid
            test_output: Test output grid (for computing loss)
            
        Returns:
            (output_logits, loss)
        """
        device = test_input.device
        
        # 1. Encode each training example
        patterns = []
        for inp, out in zip(train_inputs, train_outputs):
            inp = inp.unsqueeze(0).to(device)  # [1, H, W]
            out = out.unsqueeze(0).to(device)  # [1, H, W]
            pattern = self.example_encoder(inp, out)  # [1, D]
            patterns.append(pattern)
        
        # Stack patterns: [1, N, D] where N = number of examples
        patterns = torch.stack(patterns, dim=1)  # [1, N, D]
        
        # 2. Aggregate patterns
        pattern = self.pattern_aggregator(patterns)  # [1, D]
        
        # 3. Encode test input
        test_input = test_input.unsqueeze(0)  # [1, H, W]
        test_emb = self.test_encoder(test_input)  # [1, H, W, D]
        
        # 4. Apply pattern
        reasoned = self.reasoner(test_emb, pattern)  # [1, H, W, D]
        
        # 5. Decode
        logits = self.decoder(reasoned)  # [1, H, W, num_colors]
        
        # Compute loss if target provided
        loss = None
        if test_output is not None:
            test_output = test_output.unsqueeze(0).to(device)
            loss = F.cross_entropy(
                logits.view(-1, self.num_colors),
                test_output.view(-1).long(),
            )
        
        return logits.squeeze(0), loss
    
    def predict(
        self,
        train_inputs: List[torch.Tensor],
        train_outputs: List[torch.Tensor],
        test_input: torch.Tensor,
    ) -> torch.Tensor:
        """Predict output grid."""
        logits, _ = self.forward(train_inputs, train_outputs, test_input)
        return logits.argmax(dim=-1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Testing
if __name__ == "__main__":
    print("Testing FewShotARC...")
    
    model = FewShotARC(embed_dim=128, num_layers=4)
    print(f"Parameters: {model.count_parameters():,}")
    
    # Simulate ARC task
    train_inputs = [torch.randint(0, 10, (5, 5)) for _ in range(3)]
    train_outputs = [torch.randint(0, 10, (5, 5)) for _ in range(3)]
    test_input = torch.randint(0, 10, (5, 5))
    test_output = torch.randint(0, 10, (5, 5))
    
    logits, loss = model(train_inputs, train_outputs, test_input, test_output)
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    pred = model.predict(train_inputs, train_outputs, test_input)
    print(f"Prediction shape: {pred.shape}")
    
    print("\n✓ FewShotARC working!")
