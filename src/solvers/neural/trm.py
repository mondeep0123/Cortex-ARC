"""
Tiny Recursive Model (TRM) for ARC-AGI.

Based on the approach that achieved ~8% on ARC-AGI-2 with only ~7M parameters.
Key idea: Use test-time training (TTT) to adapt to each task.

Reference: ARC Prize 2025 winning solutions.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np

from ..base import Solver, SolverResult
from ...core.grid import Grid
from ...core.task import Task


class TRMBlock(nn.Module):
    """Single block of the Tiny Recursive Model."""
    
    def __init__(self, hidden_dim: int = 256, num_colors: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors
        
        # Convolutional layers for local patterns
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        # Layer norm for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, H, W, hidden_dim) tensor
        
        Returns:
            (B, H, W, hidden_dim) tensor
        """
        B, H, W, D = x.shape
        
        # Reshape for conv: (B, D, H, W)
        h = x.permute(0, 3, 1, 2)
        
        # Conv block with residual
        h = F.gelu(self.conv1(h))
        h = self.conv2(h)
        
        # Back to (B, H, W, D)
        h = h.permute(0, 2, 3, 1)
        
        # Apply gating
        gate = self.gate(h)
        h = gate * h
        
        # Residual + norm
        x = self.norm1(x + h)
        
        return x


class TinyRecursiveModel(nn.Module):
    """
    Tiny Recursive Model for ARC-AGI.
    
    Key features:
    - Small model (~7M params)
    - Test-time training on each task
    - Recursive application with iteration
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        num_colors: int = 10,
        max_size: int = 30,
        num_iterations: int = 8
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors
        self.max_size = max_size
        self.num_iterations = num_iterations
        
        # Input embedding
        self.color_embed = nn.Embedding(num_colors, hidden_dim)
        
        # Position encoding
        self.pos_h = nn.Embedding(max_size, hidden_dim // 2)
        self.pos_w = nn.Embedding(max_size, hidden_dim // 2)
        
        # Main blocks (shared across iterations)
        self.blocks = nn.ModuleList([
            TRMBlock(hidden_dim, num_colors) for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_colors)
        
        # Iteration encoding
        self.iter_embed = nn.Embedding(num_iterations, hidden_dim)
    
    def embed_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Embed a grid tensor.
        
        Args:
            grid: (B, H, W) integer tensor
        
        Returns:
            (B, H, W, hidden_dim) embedding
        """
        B, H, W = grid.shape
        
        # Color embedding
        x = self.color_embed(grid)  # (B, H, W, D)
        
        # Position encoding
        h_pos = torch.arange(H, device=grid.device)
        w_pos = torch.arange(W, device=grid.device)
        
        h_emb = self.pos_h(h_pos)  # (H, D/2)
        w_emb = self.pos_w(w_pos)  # (W, D/2)
        
        # Combine positions
        pos = torch.cat([
            h_emb.unsqueeze(1).expand(-1, W, -1),
            w_emb.unsqueeze(0).expand(H, -1, -1)
        ], dim=-1)  # (H, W, D)
        
        x = x + pos.unsqueeze(0)  # Broadcast over batch
        
        return x
    
    def forward(
        self,
        input_grid: torch.Tensor,
        num_iterations: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass with recursive iterations.
        
        Args:
            input_grid: (B, H, W) integer tensor
            num_iterations: Override default iterations
        
        Returns:
            (B, H, W, num_colors) logits
        """
        num_iterations = num_iterations or self.num_iterations
        
        # Initial embedding
        x = self.embed_grid(input_grid)
        
        # Recursive iterations
        for iter_idx in range(num_iterations):
            # Add iteration embedding
            iter_emb = self.iter_embed(
                torch.tensor([iter_idx], device=x.device)
            )
            x = x + iter_emb.unsqueeze(1).unsqueeze(1)
            
            # Apply blocks
            for block in self.blocks:
                x = block(x)
        
        # Output projection
        logits = self.output_proj(x)  # (B, H, W, num_colors)
        
        return logits
    
    def predict(self, input_grid: torch.Tensor) -> torch.Tensor:
        """
        Make a prediction.
        
        Args:
            input_grid: (B, H, W) integer tensor
        
        Returns:
            (B, H, W) integer tensor
        """
        logits = self.forward(input_grid)
        return logits.argmax(dim=-1)


class TRMSolver(Solver):
    """
    Solver wrapper for TRM with test-time training.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        ttt_steps: int = 100,
        ttt_lr: float = 1e-3,
        device: str = "cuda",
        name: str = "trm"
    ):
        super().__init__(name=name)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.ttt_steps = ttt_steps
        self.ttt_lr = ttt_lr
        
        # Create model
        self.model = TinyRecursiveModel(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks
        ).to(self.device)
    
    def _grid_to_tensor(self, grid: Grid) -> torch.Tensor:
        """Convert Grid to tensor."""
        return torch.tensor(grid.data, dtype=torch.long, device=self.device)
    
    def _tensor_to_grid(self, tensor: torch.Tensor) -> Grid:
        """Convert tensor to Grid."""
        return Grid(data=tensor.cpu().numpy().astype(np.int8))
    
    def _test_time_train(self, task: Task):
        """
        Perform test-time training on the task's training examples.
        """
        # Reset model for new task (optional: could keep pretrained weights)
        # For now, we'll fine-tune on each task
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.ttt_lr)
        
        # Prepare training data
        inputs = [self._grid_to_tensor(p.input) for p in task.train]
        outputs = [self._grid_to_tensor(p.output) for p in task.train]
        
        # Pad to same size for batching
        max_h = max(g.shape[0] for g in inputs + outputs)
        max_w = max(g.shape[1] for g in inputs + outputs)
        
        def pad_tensor(t, h, w):
            padded = torch.zeros(h, w, dtype=t.dtype, device=t.device)
            padded[:t.shape[0], :t.shape[1]] = t
            return padded
        
        inputs = torch.stack([pad_tensor(t, max_h, max_w) for t in inputs])
        outputs = torch.stack([pad_tensor(t, max_h, max_w) for t in outputs])
        
        # Training loop
        self.model.train()
        for step in range(self.ttt_steps):
            optimizer.zero_grad()
            
            logits = self.model(inputs)  # (B, H, W, C)
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, 10),
                outputs.reshape(-1)
            )
            
            loss.backward()
            optimizer.step()
        
        self.model.eval()
    
    def solve(self, task: Task) -> SolverResult:
        """Solve using TRM with test-time training."""
        
        # Test-time train on this task
        self._test_time_train(task)
        
        predictions = []
        confidence = []
        
        # Predict for each test case
        with torch.no_grad():
            for test_pair in task.test:
                input_tensor = self._grid_to_tensor(test_pair.input).unsqueeze(0)
                
                # Get prediction
                pred_tensor = self.model.predict(input_tensor)[0]
                
                # Crop to expected size if needed
                h, w = test_pair.input.shape
                pred_tensor = pred_tensor[:h, :w]
                
                pred_grid = self._tensor_to_grid(pred_tensor)
                
                predictions.append([pred_grid])
                confidence.append([0.5])  # Neutral confidence
        
        return SolverResult(
            task_id=task.task_id,
            predictions=predictions,
            confidence=confidence,
            metadata={"model": "TRM", "ttt_steps": self.ttt_steps}
        )
