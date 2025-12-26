"""
Evaluate trained model on handcrafted benchmark.

This is the TRUE test - not synthetic, but carefully designed
puzzles that reflect ARC-level cognitive challenges.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from primitives.object_cognition_primitive import ObjectCognitionPrimitive
from primitives.base_primitive import PrimitiveSkillConfig
from primitives.handcrafted_benchmark import HandcraftedObjectBenchmark


def evaluate_on_handcrafted_benchmark(model_path: str):
    """Evaluate trained model on handcrafted benchmark."""
    
    print("\n" + "="*70)
    print("HANDCRAFTED BENCHMARK EVALUATION")
    print("="*70)
    print("\nThis is the REAL test of object cognition ability!")
    print("Not synthetic - each puzzle handcrafted to test specific skills\n")
    
    # Load benchmark
    print("📋 Loading handcrafted benchmark...")
    benchmark = HandcraftedObjectBenchmark()
    print(f"   {len(benchmark)} puzzles across 4 difficulty levels\n")
    
    # Load model
    print("🤖 Loading trained model...")
    config = PrimitiveSkillConfig(
        name="object_cognition",
        hidden_dim=256,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model = ObjectCognitionPrimitive(config).to(config.device)
    model.load_checkpoint(model_path)
    model.eval()
    print(f"   Model loaded from {model_path}\n")
    
    # Evaluate
    print("="*70)
    print("TESTING...")
    print("="*70 + "\n")
    
    results_by_difficulty = {'easy': [], 'medium': [], 'hard': [], 'arc': []}
    all_results = []
    
    for puzzle in benchmark:
        grid = puzzle['grid']
        expected = puzzle['expected_count']
        
        # Convert to tensor
        grid_tensor = torch.from_numpy(grid).long().unsqueeze(0).to(config.device)
        
        # Predict
        with torch.no_grad():
            output = model(grid_tensor)
            # Use regression output (primary)
            count_reg = output['count_reg'].item()
            predicted_count = round(count_reg * 15.0)
            confidence = count_reg
        
        # Check correctness
        correct = (predicted_count == expected)
        
        result = {
            'name': puzzle['name'],
            'difficulty': puzzle['difficulty'],
            'expected': expected,
            'predicted': predicted_count,
            'correct': correct,
            'confidence': confidence,
            'description': puzzle['description']
        }
        
        all_results.append(result)
        results_by_difficulty[puzzle['difficulty']].append(result)
        
        # Print result
        status = "✓" if correct else "✗"
        print(f"{status} [{puzzle['difficulty']:6s}] {puzzle['name']:20s}: "
              f"Expected {expected}, Got {predicted_count} "
              f"({confidence:.2%} conf)")
    
    # Summary
    print(f"\n{'='*70}")
    print("RESULTS BY DIFFICULTY")
    print(f"{'='*70}\n")
    
    overall_correct = 0
    overall_total = 0
    
    for diff in ['easy', 'medium', 'hard', 'arc']:
        results = results_by_difficulty[diff]
        if not results:
            continue
        
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        overall_correct += correct
        overall_total += total
        
        # Target for each level
        targets = {'easy': 1.0, 'medium': 0.8, 'hard': 0.6, 'arc': 0.4}
        target = targets[diff]
        meets_target = accuracy >= target
        status = "✅" if meets_target else "❌"
        
        print(f"{diff.upper():8s}: {correct}/{total} = {accuracy:5.1%}  "
              f"(target: ≥{target:.0%}) {status}")
    
    # Overall
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0
    print(f"\n{'─'*70}")
    print(f"OVERALL:  {overall_correct}/{overall_total} = {overall_acc:5.1%}")
    print(f"{'─'*70}\n")
    
    # Final verdict
    print("="*70)
    print("FINAL VERDICT")
    print("="*70 + "\n")
    
    if overall_acc >= 0.70:
        print("🎉 EXCELLENT! MODEL IS ARC-READY!")
        print(f"   Overall accuracy: {overall_acc:.1%} (target: ≥70%)")
        print("\n   This is a STRONG TEACHER for knowledge distillation.")
        print("   Expected student accuracy: ~" + f"{overall_acc * 0.85:.0%} to {overall_acc * 0.90:.0%}")
        print("\n   ✓ Ready to train remaining primitives!")
        verdict = "PASS"
    elif overall_acc >= 0.60:
        print("👍 GOOD! Model shows strong object cognition.")
        print(f"   Overall accuracy: {overall_acc:.1%}")
        print("\n   This is a DECENT TEACHER, but room for improvement.")
        print("   Expected student accuracy: ~" + f"{overall_acc * 0.85:.0%}")
        print("\n   Consider:")
        print("   - Larger model for higher capacity")
        print("   - More curriculum diversity")
        verdict = "DECENT"
    elif overall_acc >= 0.50:
        print("⚠️  ACCEPTABLE - Model has basic object cognition.")
        print(f"   Overall accuracy: {overall_acc:.1%}")
        print("\n   This is an OKAY TEACHER, but not ideal.")
        print("   Expected student accuracy: ~" + f"{overall_acc * 0.85:.0%}")
        print("\n   Recommend:")
        print("   - Improve curriculum task design")
        print("   - Increase model capacity")
        print("   - Train longer")
        verdict = "ACCEPTABLE"
    else:
        print("❌ NEEDS IMPROVEMENT - Object cognition is weak.")
        print(f"   Overall accuracy: {overall_acc:.1%} (target: ≥70%)")
        print("\n   Not ready for knowledge distillation yet.")
        print("\n   Action needed:")
        print("   - Redesign curriculum")
        print("   - Much larger model")
        print("   - Check for training issues")
        verdict = "FAIL"
    
    # Save detailed results
    import json
    results_file = "results/handcrafted_benchmark_results.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'overall_accuracy': overall_acc,
            'overall_correct': overall_correct,
            'overall_total': overall_total,
            'verdict': verdict,
            'by_difficulty': {
                diff: {
                    'accuracy': sum(1 for r in results if r['correct']) / len(results) if results else 0,
                    'correct': sum(1 for r in results if r['correct']),
                    'total': len(results)
                }
                for diff, results in results_by_difficulty.items()
            },
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {results_file}\n")
    
    return verdict, overall_acc, all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default to high-performance model
        model_path = "checkpoints/object_cognition_best.pt"
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  HANDCRAFTED BENCHMARK EVALUATION  ".center(68) + "█")
    print("█" + "  The TRUE test of object cognition  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    verdict, accuracy, results = evaluate_on_handcrafted_benchmark(model_path)
    
    print("="*70)
    print(f"Verdict: {verdict} ({accuracy:.1%})")
    print("="*70 + "\n")
