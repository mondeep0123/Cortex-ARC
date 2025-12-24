"""
Train Numerosity Primitive

Simple training script using existing infrastructure.
"""

import torch
from src.primitives.numerosity_primitive import NumerosityPrimitive
from src.primitives.curriculum_numerosity import create_numerosity_loaders
from src.primitives.base_primitive import PrimitiveSkillConfig
import yaml

print("="*70)
print("NUMEROSITY PRIMITIVE TRAINING")
print("="*70)
print()

# Load config
with open('configs/high_performance.yaml') as f:
    config_dict = yaml.safe_load(f)

num_config = config_dict['numerosity']

# Create primitive config
config = PrimitiveSkillConfig(
    name="numerosity",
    hidden_dim=num_config['hidden_dim'],
    num_layers=num_config['num_layers'],
    learning_rate=num_config['learning_rate'],
    batch_size=num_config['batch_size'],
    max_epochs=num_config['max_epochs'],
    patience=num_config['patience'],
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Config: {config.hidden_dim}D, {config.num_layers} layers, LR={config.learning_rate}")
print()

# Create model
model = NumerosityPrimitive(config).to(config.device)

# Load Object Cognition model if available
import os
obj_cog_path = "checkpoints/object_cognition_best.pt"
if os.path.exists(obj_cog_path):
    print(f"\n🔗 Loading Object Cognition model for feature extraction...")
    model.load_object_cognition(obj_cog_path)
    print("   Object Cognition will provide segmentation features")
else:
    print(f"\n⚠ Object Cognition checkpoint not found at {obj_cog_path}")
    print("   Using simple fallback (grid > 0) for object mask")

params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: {params:,} parameters ({trainable_params:,} trainable)")
print()

# Create curriculum
print("Generating curriculum...")
train_loader, val_loader, test_loader = create_numerosity_loaders(
    train_size=num_config['train_size'],
    val_size=num_config['val_size'],
    test_size=num_config['test_size'],
    batch_size=config.batch_size,
    grid_size_range=tuple(num_config['grid_size_range'])
)
print()

# Train
print("="*70)
print("TRAINING")
print("="*70)
print()

metrics = model.fit(train_loader, val_loader)

print()
print("="*70)
print("GENERALIZATION TEST")
print("="*70)
print()

# Load best
model.load_checkpoint(f"checkpoints/{config.name}_best.pt")

# Test
from src.primitives.base_primitive import PrimitiveEvaluator
results = PrimitiveEvaluator.evaluate_generalization(
    model, train_loader, val_loader, test_loader
)

PrimitiveEvaluator.print_report("Numerosity", results)

# Save final
model.save_checkpoint("models/numerosity_final.pt")
print("\n✓ Saved to models/numerosity_final.pt")

