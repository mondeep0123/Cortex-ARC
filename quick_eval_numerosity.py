import sys
sys.path.insert(0, 'src')
from primitives.benchmark_numerosity import PureNumerosityBenchmark
from primitives.numerosity_primitive import NumerosityPrimitive
from primitives.base_primitive import PrimitiveSkillConfig
import torch

# Load benchmark
benchmark = PureNumerosityBenchmark()

# Load model
checkpoint = torch.load('checkpoints/numerosity_best.pt', map_location='cpu')
config = checkpoint.get('config', PrimitiveSkillConfig(name='numerosity', hidden_dim=256, device='cpu'))
model = NumerosityPrimitive(config).to('cpu')
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"Testing on {len(benchmark)} handcrafted puzzles...")
print()

correct = 0
for puzzle in benchmark.puzzles:
    grid_tensor = torch.from_numpy(puzzle['grid']).long().unsqueeze(0)
    
    with torch.no_grad():
        output = model(grid_tensor)
        pred_total = int(round(output['total_count'].item() * 30))
    
    expected = puzzle['expected_total']
    tolerance = puzzle.get('tolerance', 0)
    is_correct = abs(pred_total - expected) <= tolerance
    
    status = "✓" if is_correct else "✗"
    print(f"{status} {puzzle['name']:20s}: {pred_total:2d} (exp: {expected:2d})")
    
    if is_correct:
        correct += 1

print()
print(f"Accuracy: {correct}/{len(benchmark)} = {correct/len(benchmark):.1%}")
