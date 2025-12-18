"""
Transformer-based solver for ARC-AGI.

Uses attention mechanisms to learn grid transformations.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np
import math

from ..base import Solver, SolverResult
from ...core.grid import Grid
from ...core.task import Task


class GridEmbedding(nn.Module):
    """Embed grids with color + position information."""
    
    def __init__(self, hidden_dim: int, num_colors: int = 10, max_size: int = 30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.color_embed = nn.Embedding(num_colors, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(max_size, max_size, hidden_dim) * 0.02)
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (B, H, W) integer tensor
        Returns:
            (B, H, W, D) embedding
        """
        B, H, W = grid.shape
        x = self.color_embed(grid)  # (B, H, W, D)
        x = x + self.pos_embed[:H, :W, :]  # Add positions
        return x


class CrossGridAttention(nn.Module):
    """
    Attention between input and output grids.
    
    Used to learn the transformation mapping.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-attention between grids.
        
        Args:
            query: (B, H1, W1, D) - output grid queries
            key: (B, H2, W2, D) - input grid keys
            value: (B, H2, W2, D) - input grid values
        
        Returns:
            (B, H1, W1, D) attended output
        """
        B, H1, W1, D = query.shape
        _, H2, W2, _ = key.shape
        
        # Flatten spatial dims for attention
        q = self.q_proj(query).view(B, H1*W1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(B, H2*W2, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(B, H2*W2, self.num_heads, self.head_dim)
        
        # (B, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (B, heads, H1*W1, head_dim)
        out = out.transpose(1, 2).reshape(B, H1, W1, D)
        
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with self and cross attention."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = CrossGridAttention(hidden_dim, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, D) current representation
            context: (B, H2, W2, D) context from input grid
        """
        B, H, W, D = x.shape
        
        # Self-attention (flatten spatial dims)
        x_flat = x.view(B, H*W, D)
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)
        x = x + attn_out.view(B, H, W, D)
        x = self.norm1(x)
        
        # Cross-attention with context (if provided)
        if context is not None:
            cross_out = self.cross_attn(x, context, context)
            x = x + cross_out
            x = self.norm2(x)
        
        # FFN
        x = x + self.ffn(x)
        x = self.norm3(x)
        
        return x


class GridTransformer(nn.Module):
    """
    Transformer model for ARC grid transformations.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_colors: int = 10,
        max_size: int = 30
    ):
        super().__init__()
        
        self.embedding = GridEmbedding(hidden_dim, num_colors, max_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, num_colors)
    
    def forward(
        self,
        input_grid: torch.Tensor,
        output_template: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate output grid given input.
        
        Args:
            input_grid: (B, H1, W1) input grid
            output_template: (B, H2, W2) optional template for output size
        
        Returns:
            (B, H, W, num_colors) logits
        """
        # Embed input
        input_emb = self.embedding(input_grid)
        
        # If no template, use input shape
        if output_template is not None:
            x = self.embedding(output_template)
        else:
            x = input_emb.clone()
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, context=input_emb)
        
        # Output projection
        return self.output_proj(x)


class TransformerSolver(Solver):
    """
    Solver using transformer architecture.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        ttt_steps: int = 200,
        ttt_lr: float = 1e-4,
        device: str = "cuda",
        name: str = "transformer"
    ):
        super().__init__(name=name)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.ttt_steps = ttt_steps
        self.ttt_lr = ttt_lr
        
        self.model = GridTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)
    
    def _grid_to_tensor(self, grid: Grid) -> torch.Tensor:
        return torch.tensor(grid.data, dtype=torch.long, device=self.device)
    
    def _tensor_to_grid(self, tensor: torch.Tensor) -> Grid:
        return Grid(data=tensor.cpu().numpy().astype(np.int8))
    
    def _test_time_train(self, task: Task):
        """Test-time training on task examples."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.ttt_lr)
        
        self.model.train()
        for step in range(self.ttt_steps):
            total_loss = 0
            
            for pair in task.train:
                optimizer.zero_grad()
                
                input_t = self._grid_to_tensor(pair.input).unsqueeze(0)
                output_t = self._grid_to_tensor(pair.output).unsqueeze(0)
                
                # Predict
                logits = self.model(input_t, output_t)
                
                # Loss
                loss = F.cross_entropy(
                    logits.reshape(-1, 10),
                    output_t.reshape(-1)
                )
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        self.model.eval()
    
    def solve(self, task: Task) -> SolverResult:
        """Solve using transformer with TTT."""
        self._test_time_train(task)
        
        predictions = []
        confidence = []
        
        with torch.no_grad():
            for test_pair in task.test:
                input_t = self._grid_to_tensor(test_pair.input).unsqueeze(0)
                
                # Predict with input shape
                logits = self.model(input_t)
                pred = logits.argmax(dim=-1)[0]
                
                predictions.append([self._tensor_to_grid(pred)])
                confidence.append([0.5])
        
        return SolverResult(
            task_id=task.task_id,
            predictions=predictions,
            confidence=confidence,
            metadata={"model": "transformer"}
        )
