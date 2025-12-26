import torch
import numpy as np
import matplotlib.pyplot as plt
from train_color_object_cognition_v3 import ColorCurriculumDataset, create_handcrafted_benchmark, ColorObjectCognition, ColorCognitionConfig

def visualize_grid(grid, title, ax):
    ax.imshow(grid, cmap='tab10', vmin=0, vmax=9)
    ax.set_title(title)
    ax.axis('off')

def analyze_grid_stats(samples_list, name):
    target_densities = []
    distractor_densities = []
    
    for item in samples_list:
        if isinstance(item, dict): # Benchmark test
            grid = item['grid']
            mask = item['expected']
        else: # Dataset sample
            grid, task, mask = item
            
        total = grid.size
        # Target color density
        target_densities.append(np.sum(mask) / total)
        # Distractor color density
        distractor = np.sum((grid > 0) & (mask == 0))
        distractor_densities.append(distractor / total)
        
    print(f"\nStats for {name}:")
    print(f"  Avg Target Density:    {np.mean(target_densities):.1%}")
    print(f"  Avg Distractor Density: {np.mean(distractor_densities):.1%}")
    print(f"  Max Distractor Density: {np.max(distractor_densities):.1%}")

def debug_curriculum():
    # 1. Inspect Handcrafted Benchmark
    print("Inspecting Handcrafted Benchmark...")
    tests = create_handcrafted_benchmark()
    analyze_grid_stats(tests, "Handcrafted Benchmark")
    
    # Detailed check for task 4
    t4 = [t for t in tests if t.get('task') == 4]
    if t4:
        analyze_grid_stats(t4, "Task 4 (Benchmark)")

    # 2. Inspect Curriculum Stages
    for stage in [1, 2, 3]:
        print(f"\n--- Stage {stage} Samples ---")
        ds = ColorCurriculumDataset(num_samples=100, stage=stage, seed=42)
        analyze_grid_stats(ds.samples, f"Stage {stage} (100 samples)")
        
        # Check task 4 density in curriculum
        t4_cur = [s for s in ds.samples if s[1] == 4]
        if t4_cur:
            analyze_grid_stats(t4_cur, f"Task 4 (Stage {stage})")
            
        # Visualize first 5
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f"Stage {stage} Examples", fontsize=16)
        
        for i in range(5):
            grid, task, mask = ds.samples[i]
            h, w = grid.shape
            visualize_grid(grid, f"Sample {i}\n{h}x{w} Task: {task}", axes[0, i])
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].axis('off')
            
        plt.tight_layout()
        plt.savefig(f"curriculum_stage_{stage}_debug.png")
        plt.close()

def analyze_benchmark_with_model(checkpoint_path, device='cuda'):
    print(f"\nEvaluating Model Checkpoint: {checkpoint_path}")
    config = ColorCognitionConfig()
    model = ColorObjectCognition(config).to(device)
    try:
        model.load(checkpoint_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    tests = create_handcrafted_benchmark()
    
    for t in tests:
        if t['name'] in ['mask_color_4_line', 'mask_color_8_diagonal', 'mask_color_5_single']:
            grid_tensor = torch.tensor(t['grid'], dtype=torch.long).unsqueeze(0).to(device)
            task_tensor = torch.tensor([t['task']], dtype=torch.long).to(device)
            
            with torch.no_grad():
                pred = model(grid_tensor, task_tensor).squeeze().cpu()
                
            print(f"\nDebug Test: {t['name']} (Task {t['task']})")
            print("Grid:")
            print(t['grid'])
            print("Predicted Probabilities (max):", pred.max().item())
            print("Predicted Probabilities (min):", pred.min().item())
            print("Predicted Mask (binary):")
            print((pred > 0.5).int().numpy())
            print("Expected Mask:")
            print(t['expected'].astype(int))

if __name__ == "__main__":
    # 1. Stats and Distribution
    tests = create_handcrafted_benchmark()
    print(f"\nBenchmark Difficulty Distribution...")
    
    # 2. Dataset Debug
    debug_curriculum()
    
    # 3. Model Debug
    analyze_benchmark_with_model('checkpoints/color_object_cognition_v3.pt')
