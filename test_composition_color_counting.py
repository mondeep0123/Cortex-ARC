
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Import model structures
from train_color_object_cognition_v3 import ColorObjectCognition, ColorCognitionConfig, TaskType
from train_staged import StagedCounter

def test_compositional_counting():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing Composition on Device: {device}\n")

    # 1. Load Vision Model (Color Selective Masking)
    vision_config = ColorCognitionConfig()
    vision_model = ColorObjectCognition(vision_config).to(device)
    
    vision_ckpt = 'checkpoints/color_object_cognition_v3.pt'
    if os.path.exists(vision_ckpt):
        checkpoint = torch.load(vision_ckpt, map_location=device)
        vision_model.load_state_dict(checkpoint['model'])
        print(f"✓ Vision Model Loaded: {vision_ckpt}")
    else:
        print(f"✖ Vision Model NOT found at {vision_ckpt}")
        return

    # 2. Load Counting Model (Numerosity)
    counting_model = StagedCounter(d_model_sub=64, hidden_size=128, chunk_size=4).to(device)
    counting_ckpt = 'checkpoints/staged_model.pt'
    if os.path.exists(counting_ckpt):
        checkpoint = torch.load(counting_ckpt, map_location=device)
        counting_model.load_state_dict(checkpoint['model_state'])
        print(f"✓ Counting Model Loaded: {counting_ckpt}")
    else:
        print(f"✖ Counting Model NOT found at {counting_ckpt}")
        return

    vision_model.eval()
    counting_model.eval()

    # 3. Define Test Cases (Small & Large Grids)
    test_cases = [
        # Small Grids
        {
            "name": "Small 3x3 - Count Red(2)",
            "grid": np.array([
                [2, 0, 2],
                [0, 1, 0],
                [2, 0, 2]
            ]),
            "target_color": 2,
            "expected_count": 4
        },
        {
            "name": "Small 5x5 - Count Blue(1)",
            "grid": np.array([
                [1, 1, 0, 0, 0],
                [1, 1, 0, 3, 0],
                [0, 0, 0, 3, 0],
                [0, 3, 3, 3, 0],
                [0, 0, 0, 0, 1]
            ]),
            "target_color": 1,
            "expected_count": 5
        },
        # Large Grids
        {
            "name": "Large 20x20 - Count Green(3)",
            "grid": (lambda: (
                g := np.zeros((20, 20), dtype=int),
                [g.__setitem__((i, i), 3) for i in range(12)],
                [g.__setitem__((np.random.randint(0, 20), np.random.randint(0, 20)), 4) for _ in range(50)],
                g
            )[-1])(),
            "target_color": 3,
            "expected_count": 12
        },
        {
            "name": "Adversarial 30x30 - Count Pink(6)",
            "grid": (lambda: (
                g := np.zeros((30, 30), dtype=int),
                # Ensure no accidental extra Pink by setting everything to 7 first
                g.fill(7),
                # Place 25 Pink pixels
                [g.__setitem__((i // 5, i % 5), 6) for i in range(25)],
                g
            )[-1])(),
            "target_color": 6,
            "expected_count": 25
        }
    ]

    print("\n" + "="*60)
    print("RUNNING COMPOSITIONAL TESTS (Vision + Counting)")
    print("="*60)

    for tc in test_cases:
        grid = tc['grid']
        target_color = tc['target_color']
        expected = tc['expected_count']
        
        # Step 1: Vision - Mask by Color
        grid_tensor = torch.from_numpy(grid).long().unsqueeze(0).to(device)
        task_tensor = torch.tensor([target_color], device=device)
        
        with torch.no_grad():
            logits = vision_model(grid_tensor, task_tensor)
            mask = (torch.sigmoid(logits) > 0.5).float()
            
            # Step 2: Counting - Count Non-Zero in Mask
            # The counter expects (grid, mask). We pass the mask as both if we just want to count the mask pixels.
            # actually if we pass (mask, mask) it counts pixels in the mask.
            # but staged counter uses (grid, mask) to extract row values.
            # if we want to count pixels in the mask regardless of color, we can just use (mask, mask).
            count_pred = counting_model(mask.squeeze(1), mask.squeeze(1))
            count = int(torch.round(count_pred).item())

        status = "✓" if count == expected else "✖"
        print(f"{status} {tc['name']:<35} | Target Color: {target_color} | Expected: {expected:<2} | Pred: {count:<2}")
        
        if count != expected:
            # Let's see if the mask was wrong or the counter
            actual_mask_count = int(mask.sum().item())
            print(f"   [Debug] Actual pixels in mask: {actual_mask_count}")

    print("="*60)

if __name__ == "__main__":
    test_compositional_counting()
