"""
High-Performance Training Script
Trains primitives with maximum accuracy target (90%+)
"""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from primitives.base_primitive import PrimitiveSkillConfig, PrimitiveEvaluator
from primitives.object_cognition_primitive import ObjectCognitionPrimitive
from primitives.curriculum_object_cognition import create_curriculum_loaders


def load_config(config_path="configs/high_performance.yaml", primitive_name="object_cognition"):
    """Load high-performance configuration."""
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs[primitive_name]


def train_high_performance_primitive(primitive_name="object_cognition"):
    """Train with high-performance configuration for maximum accuracy."""
    
    print("\n" + "🚀"*35)
    print("🚀" + " "*68 + "🚀")
    print("🚀" + f"  HIGH-PERFORMANCE TRAINING: {primitive_name.upper()}  ".center(68) + "🚀")
    print("🚀" + "  TARGET: 90%+ Test Accuracy (Strong Teacher!)  ".center(68) + "🚀")
    print("🚀" + " "*68 + "🚀")
    print("🚀"*35 + "\n")
    
    # Load high-performance config
    hp_config = load_config(primitive_name=primitive_name)
    
    # Create config object
    config = PrimitiveSkillConfig(
        name=primitive_name,
        hidden_dim=hp_config['hidden_dim'],
        num_layers=hp_config['num_layers'],
        learning_rate=hp_config['learning_rate'],
        batch_size=hp_config['batch_size'],
        max_epochs=hp_config['max_epochs'],
        patience=hp_config['patience'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"🔧 Configuration:")
    print(f"   Model:")
    print(f"     Hidden dim: {config.hidden_dim} (high capacity)")
    print(f"     Layers: {config.num_layers}")
    print(f"     Parameters: ~{config.hidden_dim * config.hidden_dim * 4:,}")
    print(f"\n   Training:")
    print(f"     Learning rate: {config.learning_rate} → {1e-6} (cosine decay)")
    print(f"     Batch size: {config.batch_size}")
    print(f"     Max epochs: {config.max_epochs}")
    print(f"     Patience: {config.patience}")
    print(f"     Device: {config.device}")
    print(f"\n   Curriculum:")
    print(f"     Train: {hp_config['train_size']:,} tasks")
    print(f"     Val: {hp_config['val_size']:,} tasks")
    print(f"     Test: {hp_config['test_size']:,} tasks")
    print(f"\n   🎯 SUCCESS CRITERIA:")
    print(f"      Test accuracy: >{hp_config['target_accuracy']:.0%}")
    print(f"      Generalization gap: <{hp_config['max_generalization_gap']:.0%}\n")
    
    # Create model
    print(f"🏗️  Initializing high-capacity model...")
    model = ObjectCognitionPrimitive(config).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"   Memory usage: ~{total_params * 4 / 1024 / 1024 * 2:.1f} MB (with gradients)\n")
    
    # Generate curriculum
    print(f"{'='*70}")
    print("STEP 1: GENERATE LARGE-SCALE CURRICULUM")
    print(f"{'='*70}\n")
    
    train_loader, val_loader, test_loader = create_curriculum_loaders(
        train_size=hp_config['train_size'],
        val_size=hp_config['val_size'],
        test_size=hp_config['test_size'],
        batch_size=config.batch_size,
        grid_size_range=tuple(hp_config['grid_size_range'])
    )
    
    # Train
    print(f"\n{'='*70}")
    print("STEP 2: TRAIN TO CONVERGENCE (with LR scheduling)")
    print(f"{'='*70}\n")
    
    print(f"📚 Training will:")
    print(f"   • Use AdamW optimizer with weight decay")
    print(f"   • Apply cosine annealing LR schedule")
    print(f"   • Save best model based on validation loss")
    print(f"   • Stop early if no improvement for {config.patience} epochs\n")
    
    metrics = model.fit(train_loader, val_loader)
    
    # Load best model
    print(f"\n📂 Loading best model from epoch {metrics.best_epoch}...")
    model.load_checkpoint(f"checkpoints/{config.name}_best.pt")
    
    # Test generalization
    print(f"\n{'='*70}")
    print("STEP 3: COMPREHENSIVE GENERALIZATION TEST")
    print(f"{'='*70}\n")
    
    results = PrimitiveEvaluator.evaluate_generalization(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    PrimitiveEvaluator.print_report(f"{primitive_name} (High-Performance)", results)
    
    # Save results
    import json
    results_path = f"results/{config.name}_hp_generalization.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            'config': hp_config,
            'model': {
                'total_parameters': total_params,
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers
            },
            'training': {
                'best_epoch': metrics.best_epoch,
                'final_val_accuracy': metrics.final_val_accuracy,
                'generalization_gap': metrics.generalization_gap,
                'is_overfitting': metrics.is_overfitting
            },
            'generalization': results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    # Final decision
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}\n")
    
    target_accuracy = hp_config['target_accuracy']
    max_gap = hp_config['max_generalization_gap']
    
    meets_accuracy = results['test_accuracy'] >= target_accuracy
    meets_gap = results['train_test_gap'] <= max_gap
    
    if meets_accuracy and meets_gap:
        print("✅ EXCELLENT! TEACHER IS READY FOR DISTILLATION")
        print(f"   Test accuracy: {results['test_accuracy']:.2%} (target: >{target_accuracy:.0%}) ✓")
        print(f"   Train-test gap: {results['train_test_gap']:.2%} (target: <{max_gap:.0%}) ✓")
        print("\nThis is a STRONG TEACHER that will produce a strong student!")
        print("Expected student accuracy after distillation: ~", end="")
        print(f"{results['test_accuracy'] * 0.85:.0%} to {results['test_accuracy'] * 0.90:.0%}")
        
        # Save final model
        final_path = f"models/{config.name}_hp_final.pt"
        model.save_checkpoint(final_path)
        print(f"\n✓ Final high-performance model saved to {final_path}")
        
        return True, model, results
        
    else:
        print("⚠️  CLOSE, BUT NOT QUITE THERE YET")
        print(f"\n   Test accuracy: {results['test_accuracy']:.2%} (target: >{target_accuracy:.0%})", 
              "✓" if meets_accuracy else "❌")
        print(f"   Train-test gap: {results['train_test_gap']:.2%} (target: <{max_gap:.0%})",
              "✓" if meets_gap else "❌")
        
        print("\n💡 Suggestions:")
        if not meets_accuracy:
            print("   • Increase hidden_dim (currently {})".format(config.hidden_dim))
            print("   • Train longer (currently max {} epochs)".format(config.max_epochs))
            print("  • Improve curriculum diversity")
        if not meets_gap:
            print("   • Increase regularization (weight_decay)")
            print("   • Add dropout")
            print("   • Generate more training data")
        
        return False, model, results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HIGH-PERFORMANCE PRIMITIVE TRAINING")
    print("Goal: 90%+ teacher accuracy for strong knowledge distillation")
    print("="*70)
    
    success, model, results = train_high_performance_primitive("object_cognition")
    
    print(f"\n{'='*70}")
    print("TRAINING SESSION COMPLETE")
    print(f"{'='*70}\n")
    
    if success:
        print("🎉 TEACHER MODEL IS BATTLE-READY!")
        print("\n📋 Next steps:")
        print("   1. Train remaining 4 primitives with high-performance config")
        print("   2. Ensure all achieve 90%+ test accuracy")
        print("   3. Distill all 5 into unified student model")
        print("   4. Evaluate student on ARC benchmark")
        print("\nExpected final student performance: 60-75% on ARC eval set")
    else:
        print("📊 REVIEW AND ITERATE")
        print("\nCheck results/object_cognition_hp_generalization.json for details")
        print("Adjust configuration in configs/high_performance.yaml")
        print("Then retrain!")
