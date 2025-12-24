"""
Evaluate Object Cognition on PURE benchmark.

Tests ONLY object detection/segmentation using IoU metric.
NO counting required!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from primitives.object_cognition_primitive import ObjectCognitionPrimitive
from primitives.base_primitive import PrimitiveSkillConfig
from primitives.pure_object_benchmark import PureObjectCognitionBenchmark


def compute_iou(pred_mask, true_mask):
    """
    Compute Intersection over Union.
    
    Args:
        pred_mask: [H, W] predicted binary mask
        true_mask: [H, W] true binary mask
    
    Returns:
        float: IoU score (0-1)
    """
    # Convert to binary
    pred_binary = (pred_mask > 0.5).astype(float)
    true_binary = (true_mask > 0.5).astype(float)
    
    # Intersection and union
    intersection = np.sum(pred_binary * true_binary)
    union = np.sum((pred_binary + true_binary) > 0)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def evaluate_pure_object_cognition(model_path: str):
    """Evaluate on pure object cognition benchmark."""
    
    print("\n" + "="*70)
    print("PURE OBJECT COGNITION EVALUATION")
    print("="*70)
    print("\nTests ONLY segmentation/detection - NO counting!")
    print("Metric: IoU (Intersection over Union)\n")
    
    # Load benchmark
    print("📋 Loading pure object cognition benchmark...")
    benchmark = PureObjectCognitionBenchmark()
    print(f"   {len(benchmark)} puzzles across 4 difficulty levels\n")
    
    # Load model
    print("🤖 Loading trained model...")
    
    # First load checkpoint to get the config
    checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use config from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"   Using config from checkpoint: hidden_dim={config.hidden_dim}")
    else:
        # Fallback to default
        config = PrimitiveSkillConfig(
            name="object_cognition",
            hidden_dim=256,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    # Create model with correct config
    model = ObjectCognitionPrimitive(config).to(config.device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"   Model loaded from {model_path}\n")
    
    # Evaluate
    print("="*70)
    print("TESTING OBJECT SEGMENTATION...")
    print("="*70 + "\n")
    
    results_by_difficulty = {'easy': [], 'medium': [], 'hard': [], 'arc': []}
    all_results = []
    
    for puzzle in benchmark:
        grid = puzzle['grid']
        expected_mask = puzzle['expected_mask']
        target_iou = puzzle['success_iou']
        
        # Convert to tensor
        grid_tensor = torch.from_numpy(grid).long().unsqueeze(0).to(config.device)
        
        # Predict mask
        with torch.no_grad():
            output = model(grid_tensor)
            pred_mask = output['segmentation'].squeeze().cpu().numpy()  # [H, W]
        
        # Compute IoU
        iou = compute_iou(pred_mask, expected_mask)
        passes = iou >= target_iou
        
        result = {
            'name': puzzle['name'],
            'difficulty': puzzle['difficulty'],
            'iou': float(iou),  # Convert to Python float
            'target_iou': float(target_iou),
            'passes': bool(passes),  # Convert to Python bool
            'description': puzzle['description']
        }
        
        all_results.append(result)
        results_by_difficulty[puzzle['difficulty']].append(result)
        
        # Print result
        status = "✓" if passes else "✗"
        print(f"{status} [{puzzle['difficulty']:6s}] {puzzle['name']:25s}: "
              f"IoU={iou:.3f} (target: ≥{target_iou:.2f}) "
              f"{'PASS' if passes else 'FAIL'}")
    
    # Summary
    print(f"\n{'='*70}")
    print("RESULTS BY DIFFICULTY")
    print(f"{'='*70}\n")
    
    overall_iou = 0.0
    overall_pass = 0
    overall_total = 0
    
    for diff in ['easy', 'medium', 'hard', 'arc']:
        results = results_by_difficulty[diff]
        if not results:
            continue
        
        avg_iou = np.mean([r['iou'] for r in results])
        passed = sum(1 for r in results if r['passes'])
        total = len(results)
        pass_rate = passed / total if total > 0 else 0
        
        overall_iou += avg_iou * total
        overall_pass += passed
        overall_total += total
        
        # Target for each level
        targets = {'easy': 1.0, 'medium': 1.0, 'hard': 1.0, 'arc': 0.9}
        target = targets[diff]
        meets_target = avg_iou >= target
        status = "✅" if meets_target else "❌"
        
        print(f"{diff.upper():8s}: {passed}/{total} pass, "
              f"Avg IoU={avg_iou:5.1%}  (target: ≥{target:.0%}) {status}")
    
    # Overall
    overall_avg_iou = overall_iou / overall_total if overall_total > 0 else 0
    overall_pass_rate = overall_pass / overall_total if overall_total > 0 else 0
    
    print(f"\n{'─'*70}")
    print(f"OVERALL:  {overall_pass}/{overall_total} pass ({overall_pass_rate:.1%})")
    print(f"          Average IoU: {overall_avg_iou:.1%}")
    print(f"{'─'*70}\n")
    
    # Final verdict
    print("="*70)
    print("FINAL VERDICT")
    print("="*70 + "\n")
    
    if overall_avg_iou >= 0.95:
        print("🎉 EXCELLENT! Object Cognition is STRONG!")
        print(f"   Average IoU: {overall_avg_iou:.1%} (target: ≥95%)")
        print(f"   Pass rate: {overall_pass_rate:.1%}")
        print("\n   This primitive provides excellent object detection/segmentation.")
        print("   Ready for:")
        print("   ✓ Composition with other primitives")
        print("   ✓ Knowledge distillation")
        print("   ✓ ARC task solving")
        verdict = "EXCELLENT"
    elif overall_avg_iou >= 0.85:
        print("👍 GOOD! Object Cognition works well.")
        print(f"   Average IoU: {overall_avg_iou:.1%}")
        print(f"   Pass rate: {overall_pass_rate:.1%}")
        print("\n   Strong segmentation ability.")
        print("   Ready for composition and distillation.")
        verdict = "GOOD"
    elif overall_avg_iou >= 0.70:
        print("⚠️  ACCEPTABLE - Object Cognition is functional.")
        print(f"   Average IoU: {overall_avg_iou:.1%}")
        print(f"   Pass rate: {overall_pass_rate:.1%}")
        print("\n   Basic segmentation works.")
        print("   May need improvement for complex ARC tasks.")
        verdict = "ACCEPTABLE"
    else:
        print("❌ NEEDS IMPROVEMENT - Segmentation is weak.")
        print(f"   Average IoU: {overall_avg_iou:.1%} (target: ≥95%)")
        print(f"   Pass rate: {overall_pass_rate:.1%}")
        print("\n   Object detection/segmentation not reliable.")
        print("   Recommend: Redesign or use algorithmic fallback.")
        verdict = "FAIL"
    
    # Save detailed results
    import json
    results_file = "results/pure_object_cognition_results.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'overall_iou': overall_avg_iou,
            'overall_pass_rate': overall_pass_rate,
            'overall_passed': overall_pass,
            'overall_total': overall_total,
            'verdict': verdict,
            'by_difficulty': {
                diff: {
                    'avg_i ou': np.mean([r['iou'] for r in results]),
                    'pass_rate': sum(1 for r in results if r['passes']) / len(results) if results else 0,
                    'total': len(results)
                }
                for diff, results in results_by_difficulty.items()
            },
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {results_file}\n")
    
    return verdict, overall_avg_iou, all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "checkpoints/object_cognition_best.pt"
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PURE OBJECT COGNITION EVALUATION  ".center(68) + "█")
    print("█" + "  Segmentation Only - No Counting!  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    verdict, iou, results = evaluate_pure_object_cognition(model_path)
    
    print("="*70)
    print(f"Verdict: {verdict} (IoU: {iou:.1%})")
    print("="*70 + "\n")
