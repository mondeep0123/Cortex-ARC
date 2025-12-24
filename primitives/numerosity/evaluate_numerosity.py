"""
Evaluate Numerosity Primitive on Handcrafted Benchmark

Tests counting accuracy on 16 handcrafted puzzles.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from primitives.numerosity.numerosity_primitive import NumerosityPrimitive
from primitives.numerosity.benchmark_numerosity import PureNumerosityBenchmark
from src.primitives.base_primitive import PrimitiveSkillConfig


def evaluate_numerosity(model_path="checkpoints/numerosity_best.pt"):
    """Evaluate numerosity model on handcrafted benchmark."""
    
    print("="*70)
    print("NUMEROSITY PRIMITIVE EVALUATION")
    print("="*70)
    print()
    
    # Load benchmark
    print("📋 Loading benchmark...")
    benchmark = PureNumerosityBenchmark()
    print(f"   {len(benchmark)} handcrafted puzzles\n")
    
    # Load model
    print("🤖 Loading trained model...")
    
    # Load checkpoint to get config
    checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"   Using config from checkpoint")
    else:
        # Fallback
        config = PrimitiveSkillConfig(
            name="numerosity",
            hidden_dim=256,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    model = NumerosityPrimitive(config).to(config.device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"   Model loaded from {model_path}\n")
    
    # Evaluate
    print("="*70)
    print("EVALUATING...")
    print("="*70)
    print()
    
    results = {
        'overall_accuracy': 0.0,
        'by_difficulty': {},
        'detailed_results': []
    }
    
    correct_total = 0
    total_puzzles = len(benchmark)
    
    # Track by difficulty
    diff_stats = {'easy': [0, 0], 'medium': [0, 0], 'hard': [0, 0], 'arc': [0, 0]}
    
    for puzzle in benchmark.puzzles:
        grid = puzzle['grid']
        expected_total = puzzle['expected_total']
        expected_max = puzzle['expected_max_color']
        tolerance = puzzle.get('tolerance', 0)
        
        # Predict
        grid_tensor = torch.from_numpy(grid).long().unsqueeze(0).to(config.device)
        
        with torch.no_grad():
            output = model(grid_tensor)
            pred_total = int(round(output['total_count'].item() * 30.0))
            pred_max = output['max_color'].argmax(dim=1).item()
        
        # Check accuracy
        total_correct = abs(pred_total - expected_total) <= tolerance
        max_correct = (pred_max == expected_max)
        
        # Overall correct if both are right
        is_correct = total_correct and max_correct
        
        if is_correct:
            correct_total += 1
            diff_stats[puzzle['difficulty']][0] += 1
        
        diff_stats[puzzle['difficulty']][1] += 1
        
        # Store detailed result
        results['detailed_results'].append({
            'name': puzzle['name'],
            'difficulty': puzzle['difficulty'],
            'expected_total': expected_total,
            'predicted_total': pred_total,
            'expected_max_color': expected_max,
            'predicted_max_color': pred_max,
            'total_correct': total_correct,
            'max_correct': max_correct,
            'passed': is_correct
        })
        
        # Print result
        status = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"{puzzle['name']:20s} | Total: {pred_total:2d} (exp: {expected_total:2d}) | "
              f"Max: {pred_max} (exp: {expected_max}) | {status}")
    
    # Calculate accuracies
    overall_accuracy = correct_total / total_puzzles
    
    print()
    print("="*70)
    print("RESULTS BY DIFFICULTY")
    print("="*70)
    
    for diff in ['easy', 'medium', 'hard', 'arc']:
        correct, total = diff_stats[diff]
        acc = (correct / total * 100) if total > 0 else 0
        results['by_difficulty'][diff] = {
            'accuracy': acc / 100.0,
            'correct': correct,
            'total': total
        }
        print(f"{diff.upper():8s}: {correct}/{total} = {acc:.1f}%")
    
    print()
    print("="*70)
    print(f"OVERALL ACCURACY: {overall_accuracy:.1%} ({correct_total}/{total_puzzles})")
    print("="*70)
    print()
    
    # Verdict
    if overall_accuracy >= 0.85:
        verdict = "EXCELLENT"
        emoji = "🎉"
    elif overall_accuracy >= 0.70:
        verdict = "GOOD"
        emoji = "✓"
    elif overall_accuracy >= 0.50:
        verdict = "FAIR"
        emoji = "⚠"
    else:
        verdict = "FAIL"
        emoji = "✗"
    
    results['overall_accuracy'] = overall_accuracy
    results['verdict'] = verdict
    
    print(f"{emoji} Verdict: {verdict} (Accuracy: {overall_accuracy:.1%})")
    print()
    
    # Save results
    output_path = "results/numerosity_benchmark_results.json"
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/numerosity_best.pt"
    evaluate_numerosity(model_path)
