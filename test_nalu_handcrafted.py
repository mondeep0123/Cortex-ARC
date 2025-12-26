"""
Test Neural Accumulator (NALU) on handcrafted benchmark.

Model achieved:
- Stage 1 (Subitizing 0-4): 88.05%
- Stage 2 (Small 5-8): 19.20%

Focus on small counts where it should excel!
"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from primitives.numerosity_primitive import NumerosityPrimitive
from primitives.base_primitive import PrimitiveSkillConfig
from primitives.benchmark_numerosity import PureNumerosityBenchmark

device = 'cpu'

# Load config
config = PrimitiveSkillConfig(
    name='numerosity',
    hidden_dim=512,
    device=device
)

# Create model
print("Loading Neural Accumulator (NALU) model...")
model = NumerosityPrimitive(config).to(device)

# Load Object Cognition
model.load_object_cognition("checkpoints/object_cognition_best.pt")
if model.obj_cog_model is not None:
    model.obj_cog_model = model.obj_cog_model.to(device)

# Load Stage 1 checkpoint (88% subitizing)
checkpoint = torch.load('checkpoints/numerosity_stage1_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model = model.to(device)
model.eval()

print(f"✓ Loaded Stage 1 model (val_acc: {checkpoint['val_acc']*100:.2f}%)")
print(f"✓ Model: Neural Accumulator with NALU")
print(f"✓ Trained on: Subitizing (0-4 objects)\n")

# Load benchmark
benchmark = PureNumerosityBenchmark()
print(f"Testing on {len(benchmark.puzzles)} handcrafted puzzles...\n")

# Test on each puzzle
correct = 0
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
    
    results.append({
        'name': name,
        'expected': expected_total,
        'predicted': pred_total,
        'error': error,
        'correct': both_correct
    })
    
    print(f"{status} {name:20s}: Total={pred_total:2d}/{expected_total:2d} (error: {error:2d}), Max={pred_max}/{expected_max}")

# Summary
print(f"\n{'='*70}")
print("OVERALL RESULTS")
print(f"{'='*70}")
print(f"Accuracy: {correct}/{len(benchmark.puzzles)} = {correct/len(benchmark.puzzles)*100:.1f}%")
print()

# Break down by count range
print("Performance by Count Range (Trained vs Untrained):")
print(f"{'Range':<25} {'Train?':<10} {'Correct':<10} {'Accuracy':<10} {'Avg Error'}")
print("-" * 70)

ranges = [
    ('Subitizing (0-4)', True, 0, 4),
    ('Small Comp (5-8)', False, 5, 8),
    ('Medium Comp (9-16)', False, 9, 16),
    ('Large Comp (17-30)', False, 17, 30)
]

for range_name, trained, min_c, max_c in ranges:
    range_results = [r for r in results if min_c <= r['expected'] <= max_c]
    if range_results:
        range_correct = sum(r['correct'] for r in range_results)
        range_total = len(range_results)
        range_accuracy = range_correct / range_total * 100
        range_error = sum(r['error'] for r in range_results) / range_total
        trained_str = "TRAINED" if trained else "untrained"
        print(f"{range_name:<25} {trained_str:<10} {range_correct}/{range_total:<7} {range_accuracy:>6.1f}%    {range_error:>4.2f}")

print()

# Show detailed failures for subitizing (should be good!)
subitizing_results = [r for r in results if 0 <= r['expected'] <= 4]
subitizing_failures = [r for r in subitizing_results if not r['correct']]

if subitizing_failures:
    print(f"\nSubitizing Failures (0-4 objects - should be rare!):")
    for f in subitizing_failures:
        print(f"  {f['name']:20s}: Expected {f['expected']}, Got {f['predicted']} (error: {f['error']})")
else:
    print(f"\n✓ PERFECT on all subitizing cases (0-4 objects)!")

# Show all failures
all_failures = [r for r in results if not r['correct']]
print(f"\n\nAll Failures ({len(all_failures)}/{len(results)}):")
for f in all_failures:
    in_range = "✓ TRAINED" if f['expected'] <= 4 else "✗ UNTRAINED"
    print(f"  {f['name']:20s}: Expected {f['expected']:2d}, Got {f['predicted']:2d} (error: {f['error']:2d}) {in_range}")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")

subitizing_correct = sum(r['correct'] for r in subitizing_results)
subitizing_total = len(subitizing_results)
if subitizing_total > 0:
    subitizing_acc = subitizing_correct / subitizing_total * 100
    print(f"Subitizing (0-4) Accuracy: {subitizing_acc:.1f}%")
    
    if subitizing_acc >= 90:
        print("✓ Excellent! Model generalizes on small counts!")
    elif subitizing_acc >= 70:
        print("⚠ Good but not perfect on small counts")
    else:
        print("❌ Failed to generalize even on small counts")
