"""
FINAL BREAKTHROUGH: Vision Transformer for Counting

KEY INSIGHT: expected_count == NON-ZERO PIXELS (verified!)

Architecture:
- Vision Transformer (ViT) to process grid
- Attention over all positions  
- Learn to COUNT non-zero pixels
- Train on handcrafted-style patterns with PERFECT labels

This WILL work because:
1. ✅ Task is simple: count(grid > 0)
2. ✅ Transformer can attend to all positions
3. ✅ Perfect labels from simple rule
4. ✅ Train on handcrafted-style patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """2D positional encoding for grid."""
    def __init__(self, d_model, max_h=30, max_w=30):
        super().__init__()
        
        # Create 2D positional encodings
        pe_h = torch.zeros(max_h, d_model // 2)
        pe_w = torch.zeros(max_w, d_model // 2)
        
        position_h = torch.arange(0, max_h).unsqueeze(1).float()
        position_w = torch.arange(0, max_w).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)
        
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)
        
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)
        
    def forward(self, h, w):
        """Get positional encoding for grid of size h x w."""
        # Combine height and width encodings
        pe = torch.cat([
            self.pe_h[:h].unsqueeze(1).repeat(1, w, 1),
            self.pe_w[:w].unsqueeze(0).repeat(h, 1, 1)
        ], dim=-1)
        
        return pe  # [h, w, d_model]


class VisionTransformerCounter(nn.Module):
    """
    Vision Transformer for counting non-zero pixels.
    
    Simple and effective!
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        
        self.d_model = d_model
        
        # Embed color values (0-9)
        self.color_embed = nn.Embedding(10, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification token (like BERT [CLS])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Count prediction head
        self.count_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()  # Ensure positive
        )
        
        # Color counts head
        self.color_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 10),
            nn.ReLU()
        )
        
    def forward(self, grid):
        """
        Args:
            grid: [batch, h, w] with color indices
        
        Returns:
            total_count: [batch, 1]
            color_counts: [batch, 10]
        """
        batch_size, h, w = grid.shape
        
        # Embed colors
        embedded = self.color_embed(grid)  # [batch, h, w, d_model]
        
        # Add positional encoding
        pos_enc = self.pos_encoding(h, w)  # [h, w, d_model]
        embedded = embedded + pos_enc.unsqueeze(0)
        
        # Flatten to sequence
        sequence = embedded.view(batch_size, h * w, self.d_model)  # [batch, h*w, d_model]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        sequence = torch.cat([cls_tokens, sequence], dim=1)  # [batch, 1+h*w, d_model]
        
        # Transformer
        transformed = self.transformer(sequence)  # [batch, 1+h*w, d_model]
        
        # Use CLS token output for prediction
        cls_output = transformed[:, 0, :]  # [batch, d_model]
        
        # Predictions
        total_count = self.count_head(cls_output)  # [batch, 1]
        color_counts = self.color_head(cls_output)  # [batch, 10]
        
        return total_count, color_counts


# Test
if __name__ == "__main__":
    print("Testing Vision Transformer Counter...\n")
    
    model = VisionTransformerCounter(d_model=128, nhead=8, num_layers=4)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    
    # Test forward pass
    batch_size = 4
    h, w = 5, 5
    
    test_grid = torch.randint(0, 10, (batch_size, h, w))
    
    print(f"Input grid shape: {test_grid.shape}")
    print(f"Sample grid:\n{test_grid[0]}\n")
    
    total_count, color_counts = model(test_grid)
    
    print(f"Output total_count shape: {total_count.shape}")
    print(f"Output color_counts shape: {color_counts.shape}")
    print(f"\nSample predictions:")
    print(f"  Total count: {total_count[0].item():.2f}")
    print(f"  Actual non-zero: {(test_grid[0] > 0).sum().item()}")
    
    print(f"\n✓ Vision Transformer ready for training!")
