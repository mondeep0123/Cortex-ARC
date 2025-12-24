"""
Continue Staged Training from Stage 3 checkpoint.

Stages completed:
- Stage 1: 100.00% ✓
- Stage 2: 99.58% ✓  
- Stage 3: 94.25% ✓

Now training Stage 4: Large Compositional (17-30 objects)
"""

import sys
sys.path.insert(0, 'src')

import torch
import yaml

# Import training function
from train_numerosity_staged import train_stage
from primitives.numerosity_primitive import NumerosityPrimitive
from primitives.base_primitive import PrimitiveSkillConfig


def main():
    print("="*70)
    print("CONTINUING STAGED TRAINING FROM STAGE 3")
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
    
    config.weight_decay = num_config['weight_decay']
    config.train_size = num_config['train_size']
    config.val_size = num_config['val_size']
    config.test_size = num_config['test_size']
    
    print(f"\nConfig: {config.hidden_dim}D, LR={config.learning_rate}")
    print(f"Device: {config.device}\n")
    
    # Create model
    model = NumerosityPrimitive(config).to(config.device)
    
    # Load Object Cognition
    print("Loading Object Cognition...")
    model.load_object_cognition("checkpoints/object_cognition_best.pt")
    
    # Load Stage 3 checkpoint
    print("Loading Stage 3 checkpoint...")
    checkpoint = torch.load('checkpoints/numerosity_stage3_best.pt', map_location=config.device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"✓ Loaded Stage 3 model (val_acc: {checkpoint['val_acc']*100:.2f}%)\n")
    
    # Train Stage 4 only
    print("\nPrevious Results:")
    print("  Stage 1 (Subitizing 0-4): 100.00%")
    print("  Stage 2 (Small Comp 5-8): 99.58%")
    print("  Stage 3 (Medium Comp 9-16): 94.25%")
    print("\nNow training Stage 4...\n")
    
    best_acc = train_stage(model, stage=4, config=config, epochs_per_stage=30)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print("Stage 1 (Subitizing 0-4):      100.00%")
    print("Stage 2 (Small Comp 5-8):       99.58%")
    print("Stage 3 (Medium Comp 9-16):     94.25%")
    print(f"Stage 4 (Large Comp 17-30):    {best_acc*100:>6.2f}%")
    print("\n✓ All stages complete!")
    print(f"✓ Final model saved to checkpoints/numerosity_stage4_best.pt")


if __name__ == "__main__":
   main()
