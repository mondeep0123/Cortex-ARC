"""
Analyze the 3 failing handcrafted tests to understand what went wrong.
Then create comprehensive evaluation with composition.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from train_color_object_cognition_v3 import ColorObjectCognition, ColorCognitionConfig

def analyze_failures():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    config = ColorCognitionConfig(device=device)
    model = ColorObjectCognition(config).to(device)
    model.load("checkpoints/color_object_cognition_v3.pt")
    model.eval()
    
    print("="*60)
    print("ANALYZING 3 FAILING HANDCRAFTED TESTS")
    print("="*60)
    
    # Test 1: mask_color_4_line (2 pixels wrong)
    print("\n--- TEST 1: mask_color_4_line ---")
    grid1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [4, 4, 4, 4, 4],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    expected1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    with torch.no_grad():
        g = torch.tensor(grid1, dtype=torch.long).unsqueeze(0).to(device)
        t = torch.tensor([4], dtype=torch.long).to(device)
        pred = model(g, t).squeeze().cpu().numpy()
    
    # Handle padding - crop to original size
    h, w = grid1.shape
    pred = pred[:h, :w]
    pred_binary = (pred > 0.5).astype(float)
    
    print(f"Grid shape: {grid1.shape}")
    print(f"Grid:\n{grid1}")
    print(f"\nExpected mask:\n{expected1}")
    print(f"\nRaw prediction (probabilities):\n{np.round(pred, 3)}")
    print(f"\nBinary prediction:\n{pred_binary.astype(int)}")
    diff = (pred_binary != expected1).astype(int)
    print(f"\nDifference (where wrong):\n{diff}")
    print(f"Pixels wrong: {diff.sum()}")
    
    # Test 2: mask_color_8_diagonal (4 pixels wrong)
    print("\n--- TEST 2: mask_color_8_diagonal ---")
    grid2 = np.array([
        [8, 0, 0, 0],
        [0, 8, 0, 0],
        [0, 0, 8, 0],
        [0, 0, 0, 8]
    ])
    expected2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    with torch.no_grad():
        g = torch.tensor(grid2, dtype=torch.long).unsqueeze(0).to(device)
        t = torch.tensor([8], dtype=torch.long).to(device)
        pred = model(g, t).squeeze().cpu().numpy()
    
    h, w = grid2.shape
    pred = pred[:h, :w]
    pred_binary = (pred > 0.5).astype(float)
    
    print(f"Grid shape: {grid2.shape}")
    print(f"Grid:\n{grid2}")
    print(f"\nExpected mask:\n{expected2}")
    print(f"\nRaw prediction (probabilities):\n{np.round(pred, 3)}")
    print(f"\nBinary prediction:\n{pred_binary.astype(int)}")
    diff = (pred_binary != expected2).astype(int)
    print(f"\nDifference (where wrong):\n{diff}")
    print(f"Pixels wrong: {diff.sum()}")
    
    # Test 3: mask_color_2_mixed (2 pixels wrong)
    print("\n--- TEST 3: mask_color_2_mixed ---")
    grid3 = np.array([
        [2, 2, 2, 0, 0, 0, 0, 0],
        [2, 2, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 3, 3],
        [0, 0, 0, 0, 0, 3, 3, 3]
    ])
    expected3 = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    with torch.no_grad():
        g = torch.tensor(grid3, dtype=torch.long).unsqueeze(0).to(device)
        t = torch.tensor([2], dtype=torch.long).to(device)
        pred = model(g, t).squeeze().cpu().numpy()
    
    h, w = grid3.shape
    pred = pred[:h, :w]
    pred_binary = (pred > 0.5).astype(float)
    
    print(f"Grid shape: {grid3.shape}")
    print(f"Grid:\n{grid3}")
    print(f"\nExpected mask:\n{expected3}")
    print(f"\nRaw prediction (probabilities):\n{np.round(pred, 3)}")
    print(f"\nBinary prediction:\n{pred_binary.astype(int)}")
    diff = (pred_binary != expected3).astype(int)
    print(f"\nDifference (where wrong):\n{diff}")
    print(f"Pixels wrong: {diff.sum()}")
    
    # DIAGNOSIS
    print("\n" + "="*60)
    print("DIAGNOSIS: Testing mask_all on these same grids")
    print("="*60)
    
    grids = [(grid1, "line"), (grid2, "diagonal"), (grid3, "mixed")]
    
    for grid, name in grids:
        with torch.no_grad():
            g = torch.tensor(grid, dtype=torch.long).unsqueeze(0).to(device)
            t = torch.tensor([0], dtype=torch.long).to(device)  # mask_all
            pred = model(g, t).squeeze().cpu().numpy()
        
        expected_all = (grid > 0).astype(float)
        h, w = expected_all.shape
        pred = pred[:h, :w]
        pred_binary = (pred > 0.5).astype(float)
        
        match = (pred_binary == expected_all).all()
        if match:
            print(f"  ✓ {name}: mask_all PASS")
        else:
            diff = (pred_binary != expected_all).sum()
            print(f"  ✗ {name}: mask_all FAIL ({int(diff)} pixels wrong)")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
The failures are likely due to:
1. Small grid sizes (4x4, 5x5) - U-Net pooling may lose info
2. Thin structures (1-pixel lines/diagonals) 
3. Curriculum didn't have enough of these edge cases

The model's curriculum IoU was 99.99% but handcrafted edge cases
reveal the curriculum didn't cover all patterns.

FIX OPTIONS:
1. Add more thin lines/diagonals to curriculum
2. Use min grid size of 8x8 for handcrafted tests (padding helps)
3. Accept minor edge case failures - still 99.99% on curriculum
""")


if __name__ == "__main__":
    analyze_failures()
