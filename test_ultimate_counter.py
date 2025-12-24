"""Test ULTIMATE counter on handcrafted benchmark."""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from train_ultimate_counter import UltimateRowByRowCounter
from primitives.benchmark_numerosity import PureNumerosityBenchmark

device = 'cpu'

print("="*70)
print("TESTING ULTIMATE COUNTER ON HANDCRAFTED BENCHMARK")
print("="*70)
print("\nLoading Ultimate Counter...")

model = UltimateRowByRowCounter(d_model=128).to(device)
checkpoint = torch.load('checkpoints/ultimate_counter_best.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"✓ Loaded (accuracy: {checkpoint['accuracy']*100:.2f}%, epoch: {checkpoint['epoch']})")
print("✓ Features: Object masking + Row-by-row + Curriculum\n")

# Test
benchmark = PureNumerosityBenchmark()
print(f"Testing on {len(benchmark.puzzles)} puzzles...\n")

correct = 0
results = []

for puzzle in benchmark.puzzles:
    grid = puzzle['grid']
    expected = puzzle['expected_total']
    
    # Create object mask (ground truth)
    obj_mask = (grid > 0).astype(np.float32)
    
    grid_tensor = torch.from_numpy(grid).long().unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(obj_mask).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_count = model(grid_tensor, mask_tensor)
        pred = int(round(pred_count.item()))
        pred = max(0, min(30, pred))
    
    status = "✓" if pred == expected else "✗"
    error = abs(pred - expected)
    
    if pred == expected:
        correct += 1
    
    results.append({'name': puzzle['name'], 'expected': expected, 'predicted': pred, 'error': error, 'correct': (pred == expected)})
    
    print(f"{status} {puzzle['name']:20s}: Got {pred:2d}, Expected {expected:2d}, Error: {error:2d}")

print(f"\n{'='*70}")
accuracy = correct / len(benchmark.puzzles) * 100
print(f"ACCURACY: {correct}/{len(benchmark.puzzles)} = {accuracy:.1f}%")
print(f"{'='*70}\n")

# By range
ranges = [
    ('0-5 (Easy)', 0, 5),
    ('6-15 (Medium)', 6, 15),
    ('16-30 (Hard)', 16, 30)
]

print("Performance by Range:")
for range_name, min_c, max_c in ranges:
    range_results = [r for r in results if min_c <= r['expected'] <= max_c]
    if range_results:
        range_correct = sum(r['correct'] for r in range_results)
        range_total = len(range_results)
        range_acc = range_correct / range_total * 100
        range_error = sum(r['error'] for r in range_results) / range_total
        print(f"  {range_name:20s}: {range_correct}/{range_total} = {range_acc:5.1f}%, Avg Error: {range_error:.2f}")

if accuracy >= 90:
    print("\n✓✓✓ BREAKTHROUGH SUCCESS!!!")
elif accuracy >= 70:
    print("\n✓✓ Major success!")
elif accuracy >= 50:
    print("\n✓ Good progress!")
else:
    print("\n⚠ Still has issues...")

print(f"\nTraining Accuracy: {checkpoint['accuracy']*100:.2f}%")
print(f"Test Accuracy: {accuracy:.1f}%")
print(f"Generalization Gap: {checkpoint['accuracy']*100 - accuracy:.1f}%")
