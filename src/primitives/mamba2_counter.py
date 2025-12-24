"""
ATTEMPT #12: MAMBA2 State Space Model for Counting

WHY MAMBA2 > TRANSFORMER FOR COUNTING:
1. Recurrent state (running accumulation!)
2. Linear complexity (efficient)
3. Sequential processing (human-like)
4. State Space Duality - bridges SSM + Attention

This WILL work because Mamba2 is designed for sequential accumulation!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimplifiedMamba2Block(nn.Module):
    """
    Simplified Mamba2 block for counting.
    
    Key idea: Maintain running state as we process grid positions sequentially.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Convolution (for local context)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters (selective state space)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)  # dt, B, C projections
        
        # State transition (A matrix)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        
        # Input projection
        x_and_res = self.in_proj(x)  # [batch, seq_len, d_inner * 2]
        x, res = x_and_res.split(self.d_inner, dim=-1)
        
        # Convolution (for context)
        x = x.transpose(1, 2)  # [batch, d_inner, seq_len]
        x = self.conv1d(x)[:, :, :seq_len]  # Trim padding
        x = x.transpose(1, 2)  # [batch, seq_len, d_inner]
        
        # Activation
        x = F.silu(x)
        
        # SSM (selective scan - simplified)
        y = self.selective_scan(x)
        
        # Gating
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, x):
        """
        Simplified selective scan operation.
        
        This is the CORE of Mamba2 - sequential state accumulation!
        """
        batch, seq_len, d_inner = x.shape
        
        # Get SSM parameters from input (selective!)
        ssm_params = self.x_proj(x)  # [batch, seq_len, d_state * 2]
        delta, B = ssm_params.split(self.d_state, dim=-1)
        
        # State space parameters
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Simplified scan (sequential accumulation)
        # In production, this would use parallel scan
        outputs = []
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device)
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, d_inner]
            delta_t = F.softplus(delta[:, t, :])  # [batch, d_state]
            B_t = B[:, t, :]  # [batch, d_state]
            
            # Update state: h' = A*h + B*x
            # Selective: use delta to control update
            h = h * torch.exp(A.unsqueeze(0) * delta_t.unsqueeze(1)) + \
                x_t.unsqueeze(-1) * B_t.unsqueeze(1)
            
            # Output: y = C*h + D*x
            y_t = (h * B_t.unsqueeze(1)).sum(dim=-1) + self.D * x_t
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [batch, seq_len, d_inner]
        
        return y


class Mamba2Counter(nn.Module):
    """
    Mamba2-based counter for numerosity.
    
    Sequential state accumulation for counting!
    """
    def __init__(self, d_model=128, n_layers=4, d_state=16):
        super().__init__()
        
        self.d_model = d_model
        
        # Color embedding
        self.color_embed = nn.Embedding(10, d_model)
        
        # Positional encoding (simple learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 900, d_model))  # Max 30x30
        
        # Mamba2 layers (sequential processing!)
        self.layers = nn.ModuleList([
            SimplifiedMamba2Block(d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Count prediction head
        self.count_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()
        )
        
        # Color counts
        self.color_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 10),
            nn.ReLU()
        )
        
    def forward(self, grid):
        """
        Args:
            grid: [batch, h, w]
        
        Returns:
            total_count, color_counts
        """
        batch, h, w = grid.shape
        seq_len = h * w
        
        # Embed colors
        embedded = self.color_embed(grid)  # [batch, h, w, d_model]
        embedded = embedded.view(batch, seq_len, self.d_model)
        
        # Add positional encoding
        if seq_len <= self.pos_embed.size(1):
            embedded = embedded + self.pos_embed[:, :seq_len, :]
        else:
            # Interpolate if grid is larger
            pos_emb = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=seq_len,
                mode='linear'
            ).transpose(1, 2)
            embedded = embedded + pos_emb
        
        # Pass through Mamba2 layers (sequential state accumulation!)
        x = embedded
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
            x = self.norm(x)
        
        # Global pooling (take last state - like RNN final hidden state)
        final_state = x[:, -1, :]  # [batch, d_model]
        
        # Predictions
        total_count = self.count_head(final_state)  # [batch, 1]
        color_counts = self.color_head(final_state)  # [batch, 10]
        
        return total_count, color_counts


# Test
if __name__ == "__main__":
    print("Testing Mamba2 Counter...\n")
    
    model = Mamba2Counter(d_model=128, n_layers=4, d_state=16)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    
    # Test
    batch_size = 4
    h, w = 5, 5
    
    test_grid = torch.randint(0, 10, (batch_size, h, w))
    
    print(f"Input: {test_grid.shape}")
    print(f"Sample:\n{test_grid[0]}\n")
    
    total_count, color_counts = model(test_grid)
    
    print(f"Output shapes:")
    print(f"  Total: {total_count.shape}")
    print(f"  Colors: {color_counts.shape}")
    print(f"\nSample prediction: {total_count[0].item():.2f}")
    print(f"Actual non-zero: {(test_grid[0] > 0).sum().item()}")
    
    print(f"\n✓ Mamba2 Counter ready! Sequential state accumulation for counting!")
