"""
Grid Decoder: Converts learned representations back to grids.

Produces output grids from the reasoning core's representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridDecoder(nn.Module):
    """
    Decodes learned representations back to color grids.
    
    Takes encoded representations and predicts color values for each cell.
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_colors: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_colors = num_colors
        
        # Project from model space to hidden
        self.hidden = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Predict color logits
        self.color_head = nn.Linear(hidden_dim, num_colors)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode representations to color predictions.
        
        Args:
            x: Encoded representations [B, H, W, embed_dim]
            
        Returns:
            Color logits [B, H, W, num_colors]
        """
        hidden = self.hidden(x)
        logits = self.color_head(hidden)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted colors (argmax of logits).
        
        Args:
            x: Encoded representations [B, H, W, embed_dim]
            
        Returns:
            Predicted colors [B, H, W]
        """
        logits = self.forward(x)
        return logits.argmax(dim=-1)


# Testing
if __name__ == "__main__":
    print("Testing GridDecoder...")
    
    decoder = GridDecoder()
    
    # Test input (simulated encoded representation)
    encoded = torch.randn(2, 5, 5, 128)  # Batch of 2, 5x5
    
    # Decode
    logits = decoder(encoded)
    predictions = decoder.predict(encoded)
    
    print(f"Input shape: {encoded.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction values range: [{predictions.min()}, {predictions.max()}]")
    
    print("\n✓ GridDecoder working!")
