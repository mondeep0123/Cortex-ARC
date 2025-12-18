"""
Training Loop for Cortex Model

Handles training phases:
- Phase 1: Color understanding
- Phase 2: Spatial reasoning (later)
- Phase 3: Pattern recognition (later)
- Phase 4: Full ARC (later)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any
import time
from pathlib import Path

from ..model import CortexModel
from .color_tasks import ColorDataLoader


class Trainer:
    """
    Trainer for Cortex model.
    
    Handles training loop, logging, checkpoints.
    """
    
    def __init__(
        self,
        model: CortexModel,
        lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training stats
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_color_phase(
        self,
        num_steps: int = 10000,
        batch_size: int = 32,
        log_every: int = 100,
        eval_every: int = 500,
        save_every: int = 1000,
    ) -> Dict[str, Any]:
        """
        Train Phase 1: Color Understanding.
        
        Args:
            num_steps: Total training steps
            batch_size: Batch size
            log_every: Log loss every N steps
            eval_every: Evaluate every N steps
            save_every: Save checkpoint every N steps
            
        Returns:
            Training statistics
        """
        print("=" * 60)
        print("PHASE 1: COLOR UNDERSTANDING")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Steps: {num_steps}")
        print(f"Batch size: {batch_size}")
        print()
        
        # Create data loader
        loader = ColorDataLoader(batch_size=batch_size, task_type="mixed")
        
        self.model.train()
        losses = []
        start_time = time.time()
        
        for step in range(num_steps):
            # Get batch
            input_grid, target_grid = loader.get_batch()
            input_grid = input_grid.to(self.device)
            target_grid = target_grid.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, loss = self.model(input_grid, target_grid)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            losses.append(loss.item())
            self.global_step += 1
            
            # Logging
            if (step + 1) % log_every == 0:
                avg_loss = sum(losses[-log_every:]) / log_every
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                
                print(f"Step {step + 1:5d}/{num_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Speed: {steps_per_sec:.1f} steps/sec")
            
            # Evaluation
            if (step + 1) % eval_every == 0:
                accuracy = self._evaluate_color()
                print(f"  → Eval accuracy: {accuracy:.2%}")
            
            # Checkpoint
            if (step + 1) % save_every == 0:
                self._save_checkpoint(f"color_step_{step + 1}.pt")
        
        # Final evaluation
        final_accuracy = self._evaluate_color()
        
        # Save final checkpoint
        self._save_checkpoint("color_final.pt")
        
        total_time = time.time() - start_time
        
        print()
        print("=" * 60)
        print(f"PHASE 1 COMPLETE")
        print(f"Final accuracy: {final_accuracy:.2%}")
        print(f"Total time: {total_time:.1f}s")
        print("=" * 60)
        
        return {
            "final_loss": losses[-1] if losses else 0,
            "final_accuracy": final_accuracy,
            "total_steps": num_steps,
            "total_time": total_time,
        }
    
    def _evaluate_color(self, num_batches: int = 10) -> float:
        """Evaluate color understanding accuracy."""
        self.model.eval()
        
        loader = ColorDataLoader(batch_size=32, task_type="mixed")
        
        total_correct = 0
        total_pixels = 0
        
        with torch.no_grad():
            for _ in range(num_batches):
                input_grid, target_grid = loader.get_batch()
                input_grid = input_grid.to(self.device)
                target_grid = target_grid.to(self.device)
                
                predictions = self.model.predict(input_grid)
                
                # Count correct predictions
                correct = (predictions == target_grid).sum().item()
                total = target_grid.numel()
                
                total_correct += correct
                total_pixels += total
        
        self.model.train()
        
        return total_correct / total_pixels if total_pixels > 0 else 0
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }, path)
        print(f"  → Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        print(f"Loaded checkpoint from {path}")


# Quick training script
if __name__ == "__main__":
    print("Testing Trainer...")
    
    # Create model
    model = CortexModel(
        num_colors=10,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
    )
    
    # Create trainer
    trainer = Trainer(model, lr=1e-3)
    
    # Quick test
    stats = trainer.train_color_phase(
        num_steps=100,
        batch_size=16,
        log_every=20,
        eval_every=50,
    )
    
    print(f"\nStats: {stats}")
