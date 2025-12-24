"""
Base class for primitive skill models.

Each primitive (Object Cognition, Numerosity, Geometry, Topology, Physics)
gets its own small, specialized neural network that learns ONE skill well.

After training and validating generalization, we distill all primitives
into a single student model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import json


@dataclass
class PrimitiveSkillConfig:
    """Configuration for a primitive skill model."""
    name: str
    hidden_dim: int = 64  # Small model!
    num_layers: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10  # Early stopping
    device: str = 'cpu'


@dataclass
class TrainingMetrics:
    """Metrics from training a primitive."""
    train_losses: List[float]
    val_losses: List[float]
    train_accuracies: List[float]
    val_accuracies: List[float]
    best_epoch: int
    final_val_accuracy: float
    generalization_gap: float  # train_acc - val_acc
    is_overfitting: bool


class PrimitiveEncoder(nn.Module):
    """
    Small encoder for grid inputs.
    Shared across all primitives but lightweight.
    """
    
    def __init__(self, hidden_dim=64, num_colors=10):
        super().__init__()
        
        # Color embedding
        self.color_embed = nn.Embedding(num_colors, 8)
        
        # Small CNN
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global pooling + projection
        self.fc = nn.Linear(32, hidden_dim)
        
    def forward(self, grid_tensor):
        """Encode [batch, h, w] → [batch, hidden_dim]"""
        batch_size, h, w = grid_tensor.shape
        
        # Embed: [batch, h, w, 8]
        x = self.color_embed(grid_tensor)
        x = x.permute(0, 3, 1, 2)  # [batch, 8, h, w]
        
        # Convolve
        x = F.relu(self.conv1(x))  # [batch, 16, h, w]
        x = self.pool(x)            # [batch, 16, h/2, w/2]
        x = F.relu(self.conv2(x))  # [batch, 32, h/2, w/2]
        
        # Global average pooling
        x = x.mean(dim=[2, 3])  # [batch, 32]
        
        # Project to hidden dim
        x = self.fc(x)  # [batch, hidden_dim]
        
        return x


class BasePrimitiveModel(nn.Module, ABC):
    """
    Base class for all primitive skill models.
    
    Each primitive:
    1. Takes a grid input
    2. Produces a specific output (depends on skill)
    3. Can be trained on curriculum tasks
    4. Can be evaluated for generalization
    """
    
    def __init__(self, config: PrimitiveSkillConfig):
        super().__init__()
        self.config = config
        self.encoder = PrimitiveEncoder(hidden_dim=config.hidden_dim)
        
        # Training state
        self.optimizer = None
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    @abstractmethod
    def forward(self, grid_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - specific to each primitive.
        
        Args:
            grid_tensor: [batch, height, width] of color indices
        
        Returns:
            Primitive-specific output (e.g., object masks, counts, etc.)
        """
        pass
    
    @abstractmethod
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss - specific to each primitive."""
        pass
    
    @abstractmethod
    def evaluate_output(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Evaluate accuracy - specific to each primitive."""
        pass
    
    def setup_training(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(  # AdamW for better generalization
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4,  # Stronger regularization
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing learning rate scheduler
        # Helps model converge to better solutions
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs,
            eta_min=1e-6  # Minimum learning rate
        )
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(self.config.device)
            # Move each tensor in targets dict to device
            targets = {k: v.to(self.config.device) for k, v in targets.items()}
            
            # Forward
            outputs = self(inputs)
            loss = self.compute_loss(outputs, targets)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_acc += self.evaluate_output(outputs, targets)
            num_batches += 1
        
        return total_loss / num_batches, total_acc / num_batches
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate on held-out data."""
        self.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.config.device)
                # Move each tensor in targets dict to device
                targets = {k: v.to(self.config.device) for k, v in targets.items()}
                
                outputs = self(inputs)
                loss = self.compute_loss(outputs, targets)
                
                total_loss += loss.item()
                total_acc += self.evaluate_output(outputs, targets)
                num_batches += 1
        
        return total_loss / num_batches, total_acc / num_batches
    
    def fit(self, train_loader, val_loader) -> TrainingMetrics:
        """
        Train the primitive model with early stopping.
        
        Returns metrics to check for overfitting.
        """
        if self.optimizer is None:
            self.setup_training()
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_epoch = 0
        
        print(f"\nTraining {self.config.name}...")
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train Acc':<12} {'Val Acc':<12} {'Status'}")
        print("-" * 70)
        
        for epoch in range(self.config.max_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Step learning rate scheduler
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.config.learning_rate
            
            # Early stopping check
            status = ""
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                best_epoch = epoch
                status = "✓ BEST"
                # Save best model
                self.save_checkpoint(f"checkpoints/{self.config.name}_best.pt")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.config.patience:
                    status = "⚠ EARLY STOP"
                    print(f"{epoch:<6} {train_loss:<12.4f} {val_loss:<12.4f} "
                          f"{train_acc:<12.2%} {val_acc:<12.2%} {status}")
                    break
            
            # Perfect model! Stop training
            if val_acc >= 0.9999:  # 99.99% = perfect
                status = "🎉 PERFECT!"
                print(f"{epoch:<6} {train_loss:<12.4f} {val_loss:<12.4f} "
                      f"{train_acc:<12.2%} {val_acc:<12.2%} {status}")
                print("\n🎉 Model achieved 100% accuracy! Stopping training.")
                break
            
            if epoch % 5 == 0 or status:
                print(f"{epoch:<6} {train_loss:<12.4f} {val_loss:<12.4f} "
                      f"{train_acc:<12.2%} {val_acc:<12.2%} {status}")
        
        # Final metrics
        final_train_acc = train_accs[best_epoch]
        final_val_acc = val_accs[best_epoch]
        gen_gap = final_train_acc - final_val_acc
        is_overfit = gen_gap > 0.15  # >15% gap = overfitting
        
        print(f"\n{'='*70}")
        print(f"Training Complete:")
        print(f"  Best Epoch: {best_epoch}")
        print(f"  Final Val Accuracy: {final_val_acc:.2%}")
        print(f"  Generalization Gap: {gen_gap:.2%}")
        print(f"  Overfitting: {'YES ❌' if is_overfit else 'NO ✓'}")
        print(f"{'='*70}\n")
        
        return TrainingMetrics(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accuracies=train_accs,
            val_accuracies=val_accs,
            best_epoch=best_epoch,
            final_val_accuracy=final_val_acc,
            generalization_gap=gen_gap,
            is_overfitting=is_overfit
        )
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.load_state_dict(checkpoint['model_state'])
        if self.optimizer and checkpoint['optimizer_state']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.best_val_loss = checkpoint['best_val_loss']


class PrimitiveEvaluator:
    """Evaluates a primitive on test sets to measure generalization."""
    
    @staticmethod
    def evaluate_generalization(
        model: BasePrimitiveModel,
        train_loader,
        val_loader,
        test_loader
    ) -> Dict[str, float]:
        """
        Comprehensive generalization test.
        
        Tests:
        1. Train set: Should be high (model can fit)
        2. Val set: Should be high (no overfitting)
        3. Test set: Should be high (true generalization)
        """
        results = {}
        
        # Train set (seen during training)
        _, train_acc = model.validate(train_loader)
        results['train_accuracy'] = train_acc
        
        # Val set (used for early stopping)
        _, val_acc = model.validate(val_loader)
        results['val_accuracy'] = val_acc
        
        # Test set (NEVER seen)
        _, test_acc = model.validate(test_loader)
        results['test_accuracy'] = test_acc
        
        # Generalization metrics
        results['train_val_gap'] = train_acc - val_acc
        results['val_test_gap'] = val_acc - test_acc
        results['train_test_gap'] = train_acc - test_acc
        
        # Overall assessment
        results['generalizes_well'] = (
            test_acc > 0.7 and  # >70% on unseen data
            results['train_test_gap'] < 0.2  # <20% gap
        )
        
        return results
    
    @staticmethod
    def print_report(primitive_name: str, results: Dict[str, float]):
        """Print evaluation report."""
        print(f"\n{'='*70}")
        print(f"GENERALIZATION REPORT: {primitive_name}")
        print(f"{'='*70}")
        print(f"\nAccuracies:")
        print(f"  Train Set:  {results['train_accuracy']:.2%}")
        print(f"  Val Set:    {results['val_accuracy']:.2%}")
        print(f"  Test Set:   {results['test_accuracy']:.2%}")
        print(f"\nGeneralization Gaps:")
        print(f"  Train-Val:  {results['train_val_gap']:+.2%}")
        print(f"  Val-Test:   {results['val_test_gap']:+.2%}")
        print(f"  Train-Test: {results['train_test_gap']:+.2%}")
        
        if results['generalizes_well']:
            print(f"\n✅ MODEL GENERALIZES WELL")
            print(f"   Ready for knowledge distillation!")
        else:
            print(f"\n❌ MODEL DOES NOT GENERALIZE")
            if results['test_accuracy'] < 0.7:
                print(f"   - Test accuracy too low (<70%)")
            if results['train_test_gap'] > 0.2:
                print(f"   - Overfitting detected (>20% gap)")
            print(f"   → Need more/better curriculum tasks")
        
        print(f"{'='*70}\n")
