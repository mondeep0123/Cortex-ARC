"""
Train Primitive 1: Object Cognition

This script:
1. Generates curriculum tasks
2. Trains object cognition primitive
3. Tests for generalization (NO OVERFITTING!)
4. Saves model if it generalizes well
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from primitives.base_primitive import PrimitiveSkillConfig, PrimitiveEvaluator
from primitives.object_cognition_primitive import ObjectCognitionPrimitive
from primitives.curriculum_object_cognition import create_curriculum_loaders


def train_object_cognition_primitive():
    """
    Train and evaluate object cognition primitive.
    """
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  TRAINING PRIMITIVE 1: OBJECT COGNITION  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    # Configuration
    config = PrimitiveSkillConfig(
        name="object_cognition",
        hidden_dim=64,  # Small model
        num_layers=2,
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=100,
        patience=10,  # Early stopping after 10 epochs without improvement
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Configuration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Device: {config.device}")
    
    # Create model
    print(f"\nInitializing model...")
    model = ObjectCognitionPrimitive(config).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Generate curriculum
    print(f"\n{'='*70}")
    print("STEP 1: GENERATE CURRICULUM")
    print(f"{'='*70}\n")
    
    train_loader, val_loader, test_loader = create_curriculum_loaders(
        train_size=7000,   # Train on 7K tasks
        val_size=1500,     # Validate on 1.5K
        test_size=1500,    # Test on 1.5K (never seen!)
        batch_size=config.batch_size,
        grid_size_range=(5, 15)
    )
    
    # Train
    print(f"\n{'='*70}")
    print("STEP 2: TRAIN WITH EARLY STOPPING")
    print(f"{'='*70}\n")
    
    metrics = model.fit(train_loader, val_loader)
    
    # Load best model
    print(f"\nLoading best model from epoch {metrics.best_epoch}...")
    model.load_checkpoint(f"checkpoints/{config.name}_best.pt")
    
    # Test generalization
    print(f"\n{'='*70}")
    print("STEP 3: TEST GENERALIZATION")
    print(f"{'='*70}\n")
    
    results = PrimitiveEvaluator.evaluate_generalization(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    PrimitiveEvaluator.print_report("Object Cognition", results)
    
    # Save results
    import json
    results_path = f"results/{config.name}_generalization.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'hidden_dim': config.hidden_dim,
                'learning_rate': config.learning_rate,
                'batch_size': config.batch_size
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
    print("FINAL DECISION")
    print(f"{'='*70}\n")
    
    if results['generalizes_well']:
        print("✅ PRIMITIVE IS READY FOR KNOWLEDGE DISTILLATION")
        print(f"   Test accuracy: {results['test_accuracy']:.2%}")
        print(f"   Train-test gap: {results['train_test_gap']:.2%}")
        print("\nThis model has learned GENERAL object cognition,")
        print("not task-specific patterns. Ready to teach the student model!")
        
        # Save final model
        final_path = f"models/{config.name}_final.pt"
        model.save_checkpoint(final_path)
        print(f"\n✓ Final model saved to {final_path}")
        
        return True, model, results
        
    else:
        print("❌ PRIMITIVE NEEDS IMPROVEMENT")
        print("\nPossible issues:")
        if results['test_accuracy'] < 0.7:
            print("  - Model not learning well (test acc < 70%)")
            print("  → Try: larger model, more training, better curriculum")
        if results['train_test_gap'] > 0.2:
            print("  - Overfitting detected (gap > 20%)")
            print("  → Try: more regularization, more diverse curriculum")
        
        return False, model, results


if __name__ == "__main__":
    success, model, results = train_object_cognition_primitive()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}\n")
    
    if success:
        print("🎉 PRIMITIVE 1 (Object Cognition) IS TRAINED AND VALIDATED!")
        print("\nNext steps:")
        print("  1. Train remaining primitives (Numerosity, Geometry, Topology, Physics)")
        print("  2. Test each for generalization")
        print("  3. Distill all into student model")
    else:
        print("⚠️  NEED TO IMPROVE PRIMITIVE BEFORE PROCEEDING")
        print("\nReview the generalization report above.")
