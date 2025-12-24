"""
Staged Training for Numerosity - Curriculum Learning

Train in stages:
1. Subitizing (0-4) → Should reach 99%+
2. Small Compositional (5-8) → Build on subitizing
3. Medium Compositional (9-16) → Full composition
4. Large Compositional (17-30) → Final stage

Each stage fine-tunes the previous stage's model.
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from pathlib import Path

from primitives.numerosity_primitive import NumerosityPrimitive
from primitives.base_primitive import PrimitiveSkillConfig, TrainingMetrics
from primitives.staged_curriculum_numerosity import create_staged_dataloaders


def train_stage(model, stage, config, epochs_per_stage=30):
    """Train one curriculum stage."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING STAGE {stage}")
    print(f"{'='*70}\n")
    
    # Create staged dataloaders
    train_loader, val_loader, test_loader = create_staged_dataloaders(
        stage=stage,
        batch_size=config.batch_size,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"Training for up to {epochs_per_stage} epochs...")
    print(f"Epoch  Train Loss   Val Loss     Train Acc    Val Acc      Status")
    print("-" * 70)
    
    for epoch in range(epochs_per_stage):
        # Train
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move to device
            grid = batch['grid'].to(config.device)
            targets = {
                'total_count': batch['total_count'].to(config.device),
                'color_counts': batch['color_counts'].to(config.device),
                'max_color': batch['max_color'].to(config.device)
            }
            
            # Forward
            outputs = model(grid)
            loss = model.compute_loss(outputs, targets)
            acc = model.evaluate_output(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
        
        train_loss = total_loss / num_batches
        train_acc = total_acc / num_batches
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                grid = batch['grid'].to(config.device)
                targets = {
                    'total_count': batch['total_count'].to(config.device),
                    'color_counts': batch['color_counts'].to(config.device),
                    'max_color': batch['max_color'].to(config.device)
                }
                
                outputs = model(grid)
                loss = model.compute_loss(outputs, targets)
                acc = model.evaluate_output(outputs, targets)
                
                val_loss += loss.item()
                val_acc += acc
                num_val_batches += 1
        
        val_loss /= num_val_batches
        val_acc /= num_val_batches
        
        # Check improvement
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            status = "✓ BEST"
            # Save checkpoint
            torch.save({
                'model_state': model.state_dict(),
                'stage': stage,
                'val_acc': val_acc,
                'epoch': epoch
            }, f'checkpoints/numerosity_stage{stage}_best.pt')
        else:
            patience_counter += 1
            status = ""
        
        # Print progress (not every epoch)
        if epoch == 0 or is_best or epoch % 5 == 0:
            print(f"{epoch:<6} {train_loss:>11.4f}  {val_loss:>11.4f}  {train_acc*100:>9.2f}%  {val_acc*100:>9.2f}%  {status}")
        
        # Perfect performance - advance to next stage!
        if val_acc >= 0.99:
            print(f"\n✓ Stage {stage} MASTERED at epoch {epoch}!")
            print(f"   Validation Accuracy: {val_acc*100:.2f}%")
            print(f"   Advancing to next stage...")
            break
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\n⚠ Early stopping at epoch {epoch}")
            break
    
    print(f"\nStage {stage} Complete!")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    if best_val_acc >= 0.99:
        print(f"✓ MASTERED! Ready for next stage.")
    elif best_val_acc >= 0.90:
        print(f"⚠ Good but not perfect. Continuing to next stage...")
    else:
        print(f"❌ Stage {stage} did not reach 90% accuracy.")
    print()
    
    # Load best model for next stage
    checkpoint = torch.load(f'checkpoints/numerosity_stage{stage}_best.pt')
    model.load_state_dict(checkpoint['model_state'])
    
    return best_val_acc


def main():
    print("="*70)
    print("STAGED NUMEROSITY TRAINING")
    print("="*70)
    
    # Load config
    with open('configs/high_performance.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    num_config = config_dict['numerosity']
    config = PrimitiveSkillConfig(
        name='numerosity',
        hidden_dim=num_config['hidden_dim'],
        learning_rate=num_config['learning_rate'],
        batch_size=num_config['batch_size'],
        max_epochs=num_config['max_epochs'],
        patience=num_config['patience'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add additional attributes
    config.weight_decay = num_config['weight_decay']
    config.train_size = num_config['train_size']
    config.val_size = num_config['val_size']
    config.test_size = num_config['test_size']
    
    print(f"\nConfig: {config.hidden_dim}D, {num_config['num_layers']} layers, LR={config.learning_rate}")
    print(f"Device: {config.device}\n")
    
    # Create model
    model = NumerosityPrimitive(config).to(config.device)
    
    # Load Object Cognition
    print("Loading Object Cognition...")
    model.load_object_cognition("checkpoints/object_cognition_best.pt")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} parameters ({trainable_params:,} trainable)\n")
    
    # Train each stage
    stage_results = {}
    
    for stage in [1, 2, 3, 4]:
        # Train this stage
        best_acc = train_stage(model, stage, config, epochs_per_stage=30)
        stage_results[stage] = best_acc
        
        # Summary
        print(f"\nStage {stage} Results:")
        print(f"  Best Accuracy: {best_acc*100:.2f}%")
        
        # Early exit if stage fails badly
        if best_acc < 0.70 and stage > 1:
            print(f"\n⚠ Stage {stage} did not reach 70% accuracy.")
            print(f"   Stopping staged training.")
            break
    
    # Final summary
    print("\n" + "="*70)
    print("STAGED TRAINING COMPLETE")
    print("="*70)
    
    for stage, acc in stage_results.items():
        stage_names = {
            1: "Subitizing (0-4)",
            2: "Small Comp (5-8)",
            3: "Medium Comp (9-16)",
            4: "Large Comp (17-30)"
        }
        print(f"Stage {stage} - {stage_names[stage]}: {acc*100:.2f}%")
    
    print(f"\n✓ Final model saved to checkpoints/numerosity_stage{max(stage_results.keys())}_best.pt")


if __name__ == "__main__":
    main()
