```
"""
Train Numerosity Primitive

Run from project root: python primitives/numerosity/train_numerosity.py

Uses lessons learned from Object Cognition:
✅ Spatial-preserving architecture
✅ MSE regression (not classification)
✅ Clear task definition
✅ Handcrafted test ready
"""

import torch
import yaml
from src.primitives.base_primitive import PrimitiveSkillConfig

# Import from same package
from numerosity_primitive import NumerosityPrimitive
from curriculum_numerosity import create_numerosity_loaders
from benchmark_numerosity import PureNumerosityBenchmark
import numpy as np
import json
from pathlib import Path


def train_numerosity():
    """Train numerosity primitive with high-performance settings."""
    
    print("="*70)
    print("NUMEROSITY PRIMITIVE TRAINING")
    print("="*70)
    print()
    
    # Load config
    with open('configs/high_performance.yaml') as f:
        config_dict = yaml.safe_load(f)
    
    numerosity_config = config_dict.get('numerosity', config_dict['object_cognition'])
    
    # Create config
    config = PrimitiveSkillConfig(
        name="numerosity",
        hidden_dim=numerosity_config['hidden_dim'],
        num_layers=numerosity_config['num_layers'],
        learning_rate=numerosity_config['learning_rate'],
        batch_size=numerosity_config['batch_size'],
        max_epochs=numerosity_config['max_epochs'],
        patience=numerosity_config['patience'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Configuration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Device: {config.device}")
    print()
    
    # Create model
    print("🚀 Creating Numerosity model...")
    model = NumerosityPrimitive(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print()
    
    # Create curriculum
    print("="*70)
    print("STEP 1: GENERATE CURRICULUM")
    print("="*70)
    print()
    
    train_loader, val_loader, test_loader = create_numerosity_loaders(
        train_size=numerosity_config['train_size'],
        val_size=numerosity_config['val_size'],
        test_size=numerosity_config['test_size'],
        batch_size=config.batch_size,
        grid_size_range=tuple(numerosity_config['grid_size_range'])
    )
    
    print()
    
    # Train
    print("="*70)
    print("STEP 2: TRAIN TO CONVERGENCE")
    print("="*70)
    print()
    
    print("📚 Training will:")
    print("   • Use AdamW optimizer with weight decay")
    print("   • Apply cosine annealing LR schedule")
    print("   • Save best model based on validation loss")
    print(f"   • Stop early if no improvement for {config.patience} epochs")
    print()
    
    print("   Training:")
    print()
    
    metrics = model.train_with_loaders(
        train_loader,
        val_loader,
        num_epochs=config.max_epochs
    )
    
    print()
    
    # Test generalization
    print("="*70)
    print("STEP 3: TEST GENERALIZATION")
    print("="*70)
    print()
    
    test_acc = model.test(test_loader)
    train_acc = metrics.train_accuracies[metrics.best_epoch]
    val_acc = metrics.final_val_accuracy
    
    print(f"Accuracies:")
    print(f"  Train: {train_acc:.2%}")
    print(f"  Val:   {val_acc:.2%}")
    print(f"  Test:  {test_acc:.2%}")
    print()
    
    train_test_gap = abs(train_acc - test_acc)
    print(f"Generalization gap (train-test): {train_test_gap:.2%}")
    print()
    
    # Verdict
    if test_acc >= 0.90 and train_test_gap < 0.15:
        print("✅ EXCELLENT! Model is ready for handcrafted tests!")
    elif test_acc >= 0.70:
        print("✓ GOOD! Model generalizes, ready for validation")
    else:
        print("⚠ WARNING: Model may need more training or better curriculum")
    
    print()
    
    # Save final model
    final_path = "models/numerosity_hp_final.pt"
    model.save_checkpoint(final_path)
    print(f"✓ Final model saved to {final_path}")
    print()
    
    # Evaluate on handcrafted
    print("="*70)
    print("STEP 4: EVALUATE ON HANDCRAFTED BENCHMARK")
    print("="*70)
    print()
    
    try:
        results = evaluate_numerosity("checkpoints/numerosity_best.pt")
        
        if results['overall_accuracy'] >= 0.85:
            print()
            print("🎉 SUCCESS! Numerosity primitive is EXCELLENT!")
        elif results['overall_accuracy'] >= 0.70:
            print()
            print("✓ PASS! Numerosity primitive is functional")
        else:
            print()
            print("⚠ Needs improvement on handcrafted tests")
    except Exception as e:
        print(f"⚠ Could not evaluate handcrafted: {e}")
        print("   Run evaluation manually later")
    
    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    

if __name__ == "__main__":
    train_numerosity()
