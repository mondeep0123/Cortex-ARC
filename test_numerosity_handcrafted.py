"""
Test staged numerosity model on handcrafted benchmark.

Model trained with cognitively-inspired staged curriculum:
- Stage 1: 100.00% (Subitizing 0-4)
- Stage 2: 99.58% (Small Comp 5-8)
- Stage 3: 94.25% (Medium Comp 9-16)  
- Stage 4: 88.42% (Large Comp 17-30)
"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from primitives.numerosity_primitive import NumerosityPrimitive
from primitives.base_primitive import PrimitiveSkillConfig
from primitives.benchmark_numerosity import PureNumerosityBenchmark

device = 'cpu'  # Use CPU for consistency

# Load config
config = PrimitiveSkillConfig(
    name='numerosity',
    hidden_dim=512,
    device=device
)

# Create model
print("Loading model...")
model = NumerosityPrimitive(config).to(device)

# Load Object Cognition
model.load_object_cognition("checkpoints/object_cognition_best.pt")
if model.obj_cog_model is not None:
    model.obj_cog_model = model.obj_cog_model.to(device)

# Load Stage 4 checkpoint (final model)
checkpoint = torch.load('checkpoints/numerosity_stage4_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model = model.to(device)
model.eval()

print(f"✓ Loaded Stage 4 model (val_acc: {checkpoint['val_acc']*100:.2f}%)")
print(f"✓ Model uses: Subitizing (0-4) + Compositional Counting (5-30)\n")

# Load benchmark
benchmark = PureNumerosityBenchmark()
print(f"Testing on {len(benchmark.puzzles)} handcrafted puzzles...\n")

# Test on each puzzle
correct = 0
total_error = 0
results = []

for puzzle in benchmark.puzzles:
    grid = puzzle['grid']
    name = puzzle['name']
    expected_total = puzzle['expected_total']
    expected_max = puzzle['expected_max_color']
    
    # Predict
    prediction = model.predict(grid)
    pred_total = prediction['total_count']
    pred_max = prediction['max_color']
    mode = prediction.get('mode', 'unknown')
    
    # Check correctness
    total_correct = (pred_total == expected_total)
    max_correct = (pred_max == expected_max)
    both_correct = total_correct and max_correct
    
    if both_correct:
        correct += 1
        status = "✓"
    else:
        status = "✗"
    
    error = abs(pred_total - expected_total)
    total_error += error
    
    results.append({
        'name': name,
        'expected': expected_total,
        'predicted': pred_total,
        'error': error,
        'mode': mode,
        'correct': both_correct
    })
    
    print(f"{status} {name:20s}: Total={pred_total:2d}/{expected_total:2d}, Max={pred_max}/{expected_max}, Mode={mode}")

# Summary
print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Accuracy: {correct}/{len(benchmark.puzzles)} = {correct/len(benchmark.puzzles)*100:.1f}%")
print(f"Average Error: {total_error/len(benchmark.puzzles):.2f} objects")
print()

# Break down by count range (matches training stages)
print("Performance by Count Range:")
stage_ranges = {
    'Subitizing (0-4)': (0, 4),
    'Small Comp (5-8)': (5, 8),
    'Medium Comp (9-16)': (9, 16),
    'Large Comp (17-30)': (17, 30)
}

for stage_name, (min_c, max_c) in stage_ranges.items():
    stage_results = [r for r in results if min_c <= r['expected'] <= max_c]
    if stage_results:
        stage_correct = sum(r['correct'] for r in stage_results)
        stage_total = len(stage_results)
        stage_error = sum(r['error'] for r in stage_results) / stage_total
        print(f"  {stage_name:25s}: {stage_correct}/{stage_total} = {stage_correct/stage_total*100:5.1f}%, Avg Error: {stage_error:.2f}")

print()

# Show failures
failures = [r for r in results if not r['correct']]
if failures:
    print("Failed Cases:")
    for f in failures:
        print(f"  {f['name']:20s}: Expected {f['expected']}, Got {f['predicted']} (error: {f['error']}, mode: {f['mode']})")
