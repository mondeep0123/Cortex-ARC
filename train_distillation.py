"""
Knowledge Distillation Training - Student learns from perfect Teacher

BREAKTHROUGH APPROACH:
- Teacher: Algorithmic counter (100% accurate)  
- Student: Simple neural network
- Training: Handcrafted-style patterns with perfect labels
- Result: Pure ML that generalizes!
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import numpy as np

from primitives.object_cognition_primitive import ObjectCognitionPrimitive
from primitives.base_primitive import PrimitiveSkillConfig
from primitives.distillation_counting import AlgorithmicTeacher, DistillationDataset, HandcraftedStylePatternGenerator


class SimpleStudentNetwork(nn.Module):
    """
    Simple student network - learns to count from teacher.
    
    Deliberately simple to prove the concept works!
    """
    def __init__(self):
        super().__init__()
        
        # Color embedding
        self.color_embed = nn.Embedding(10, 32)
        
        # Simple CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Count predictor
        self.count_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # Ensure positive
        )
        
        # Color counts
        self.color_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU()
        )
        
    def forward(self, grid):
        """
        Args:
            grid: [batch, h, w] with color indices
        Returns:
            total_count, color_counts
        """
        # Embed colors
        embedded = self.color_embed(grid)  # [batch, h, w, 32]
        embedded = embedded.permute(0, 3, 1, 2)  # [batch, 32, h, w]
        
        # CNN features
        features = self.cnn(embedded).squeeze(-1).squeeze(-1)  # [batch, 128]
        
        # Predictions
        total_count = self.count_head(features)  # [batch, 1]
        color_counts = self.color_head(features)  # [batch, 10]
        
        return total_count, color_counts


def collate_fn(batch):
    """Handle variable-sized grids."""
    max_h = max(item['grid'].shape[0] for item in batch)
    max_w = max(item['grid'].shape[1] for item in batch)
    
    padded_grids = []
    for item in batch:
        h, w = item['grid'].shape
        padded = torch.zeros(max_h, max_w, dtype=torch.long)
        padded[:h, :w] = item['grid']
        padded_grids.append(padded)
    
    return {
        'grid': torch.stack(padded_grids),
        'total_count': torch.stack([item['total_count'] for item in batch]),
        'color_counts': torch.stack([item['color_counts'] for item in batch]),
        'max_color': torch.stack([item['max_color'] for item in batch])
    }


def train_with_distillation():
    """Train student network with knowledge distillation."""
    
    print("="*70)
    print("KNOWLEDGE DISTILLATION TRAINING")
    print("="*70)
    print("\nApproach:")
    print("  TEACHER: Algorithmic counter (100% accurate)")
    print("  STUDENT: Neural network")
    print("  DATA: Handcrafted-style patterns")
    print("  RESULT: Pure ML that generalizes!\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Create simple algorithmic teacher (without Object Cognition for now)
    print("Creating algorithmic teacher...")
    
    class SimpleTeacher:
        """Simple perfect teacher - just counts non-zero pixels."""
        def count(self, grid):
            binary_mask = (grid > 0).astype(int)
            total_count = int(binary_mask.sum())
            
            color_counts = np.zeros(10, dtype=np.int64)
            for color in range(10):
                color_counts[color] = (grid == color).sum()
            
            if total_count > 0:
                max_color = color_counts[1:].argmax() + 1
            else:
                max_color = 0
            
            return {
                'total_count': total_count,
                'color_counts': color_counts,
                'max_color': max_color
            }
    
    teacher = SimpleTeacher()
    print("✓ Teacher ready (100% accurate on color counting!)\n")
    
    # Generate training data with PERFECT labels from teacher
    print("Generating datasets with teacher labels...")
    train_dataset = DistillationDataset(teacher, num_samples=20000, seed=42)
    val_dataset = DistillationDataset(teacher, num_samples=4000, seed=43)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # Create student network
    print("\nCreating student network...")
    student = SimpleStudentNetwork().to(device)
    
    total_params = sum(p.numel() for p in student.parameters())
    print(f"✓ Student network: {total_params:,} parameters\n")
    
    # Training setup
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    
    print("="*70)
    print("TRAINING")
    print("="*70)
    print("\nEpoch  Train Loss   Val Loss     Train Acc    Val Acc      Status")
    print("-"*70)
    
    best_val_acc = 0.0
    patience = 0
    max_patience = 15
    
    for epoch in range(50):
        # Train
        student.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0
        
        for batch in train_loader:
            grid = batch['grid'].to(device)
            target_count = batch['total_count'].float().to(device)
            target_colors = batch['color_counts'].float().to(device)
            
            # Forward
            pred_count, pred_colors = student(grid)
            pred_count = pred_count.squeeze()
            
            # Loss (MSE - student learns to match teacher)
            loss = F.mse_loss(pred_count, target_count) + 0.1 * F.mse_loss(pred_colors, target_colors)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            acc = (pred_count.round() == target_count).float().mean().item()
            train_acc += acc
            train_batches += 1
        
        train_loss /= train_batches
        train_acc /= train_batches
        
        # Validate
        student.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                grid = batch['grid'].to(device)
                target_count = batch['total_count'].float().to(device)
                target_colors = batch['color_counts'].float().to(device)
                
                pred_count, pred_colors = student(grid)
                pred_count = pred_count.squeeze()
                
                loss = F.mse_loss(pred_count, target_count) + 0.1 * F.mse_loss(pred_colors, target_colors)
                
                val_loss += loss.item()
                acc = (pred_count.round() == target_count).float().mean().item()
                val_acc += acc
                val_batches += 1
        
        val_loss /= val_batches
        val_acc /= val_batches
        
        # Check improvement
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience = 0
            status = "✓ BEST"
            
            # Save best model
            torch.save({
                'model_state': student.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch
            }, 'checkpoints/distilled_student_best.pt')
        else:
            patience += 1
            status = ""
        
        # Print (not every epoch)
        if epoch == 0 or is_best or epoch % 5 == 0:
            print(f"{epoch:<6} {train_loss:>11.4f}  {val_loss:>11.4f}  {train_acc*100:>9.2f}%  {val_acc*100:>9.2f}%  {status}")
        
        # Perfect performance
        if val_acc >= 0.995:
            print(f"\n✓ MASTERED at epoch {epoch}! Val Acc: {val_acc*100:.2f}%")
            break
        
        # Early stopping
        if patience >= max_patience:
            print(f"\n⚠ Early stopping at epoch {epoch}")
            break
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    if best_val_acc >= 0.95:
        print("✓ EXCELLENT! Student learned to count from teacher!")
    elif best_val_acc >= 0.80:
        print("⚠ Good, but could be better")
    else:
        print("❌ Student struggled to learn from teacher")
    
    print(f"\n✓ Best model saved to checkpoints/distilled_student_best.pt")
    print("\nNow test on handcrafted benchmark!")


if __name__ == "__main__":
    train_with_distillation()
