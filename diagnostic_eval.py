import torch
import numpy as np
import matplotlib.pyplot as plt
from train_color_object_cognition_v3 import ColorCurriculumDataset, create_handcrafted_benchmark, ColorObjectCognition, ColorCognitionConfig

@torch.no_grad()
def run_diagnostic(checkpoint_path, device='cuda'):
    config = ColorCognitionConfig()
    model = ColorObjectCognition(config).to(device)
    try:
        model.load(checkpoint_path)
    except:
        print("Waiting for checkpoint...")
        return
    model.eval()

    tests = create_handcrafted_benchmark()
    
    stats = {
        'total': len(tests),
        'perfect': 0,
        'precision_total': [],
        'recall_total': [],
        'empty_stats': [],
        'by_task': {}
    }

    for t in tests:
        grid = torch.tensor(t['grid'], dtype=torch.long).unsqueeze(0).to(device)
        task = t['task']
        task_tensor = torch.tensor([task], dtype=torch.long).to(device)
        expected = t['expected']
        
        logits = model(grid, task_tensor).squeeze().cpu()
        pred = torch.sigmoid(logits).numpy()
        h, w = expected.shape
        pred = pred[:h, :w]
        pred_bin = (pred > 0.5).astype(np.float32)

        is_perfect = (pred_bin == expected).all()
        if is_perfect: stats['perfect'] += 1

        tp = np.sum((pred_bin == 1) & (expected == 1))
        fp = np.sum((pred_bin == 1) & (expected == 0))
        fn = np.sum((pred_bin == 0) & (expected == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        
        is_empty = np.sum(pred_bin) == 0
        should_be_empty = np.sum(expected) == 0
        
        stats['precision_total'].append(precision)
        stats['recall_total'].append(recall)
        stats['empty_stats'].append(1.0 if (is_empty and not should_be_empty) else 0.0)

        if task not in stats['by_task']:
            stats['by_task'][task] = []
        stats['by_task'][task].append(1.0 if is_perfect else 0.0)

    print("\n" + "="*40)
    print("DIAGNOSTIC REPORT")
    print("="*40)
    print(f"Handcrafted Accuracy: {stats['perfect']}/{stats['total']} ({stats['perfect']/stats['total']:.1%})")
    print(f"P: {np.mean(stats['precision_total']):.1%} | R: {np.mean(stats['recall_total']):.1%} | Empty: {np.mean(stats['empty_stats']):.1%}")
    
    tasks_res = []
    for task_id in sorted(stats['by_task'].keys()):
        scores = stats['by_task'][task_id]
        tasks_res.append(f"T{task_id}:{np.mean(scores):.0%}")
    print("Tasks: " + " ".join(tasks_res))

    # --- CURRICULUM SELF-EVAL ---
    print("\n" + "="*40)
    print("CURRICULUM SELF-EVAL (Stage 3)")
    print("="*40)
    ds = ColorCurriculumDataset(num_samples=100, stage=3, seed=99)
    cur_perfect = 0
    cur_recalls = []
    
    for i in range(100):
        grid, task, mask = ds[i]
        grid_t = grid.unsqueeze(0).to(device)
        task_t = task.unsqueeze(0).to(device)
        expected = mask.numpy()
        
        logits = model(grid_t, task_t).squeeze().cpu()
        pred = torch.sigmoid(logits).numpy()
        h, w = expected.shape
        pred = pred[:h, :w]
        pred_bin = (pred > 0.5).astype(np.float32)
        
        if (pred_bin == expected).all(): cur_perfect += 1
        
        tp = np.sum((pred_bin == 1) & (expected == 1))
        fn = np.sum((pred_bin == 0) & (expected == 1))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        cur_recalls.append(recall)
        
    print(f"Curriculum Accuracy (Stage 3): {cur_perfect/100:.1%}")
    print(f"Curriculum Avg Recall:         {np.mean(cur_recalls):.1%}")

if __name__ == "__main__":
    run_diagnostic('checkpoints/color_object_cognition_v3.pt')
