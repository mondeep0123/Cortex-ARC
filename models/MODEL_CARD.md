# Cortex-ARC Model v4

## Performance
- ARC-AGI-1 Training: 64%
- ARC-AGI-1 Evaluation: 58%

## Configuration
- Embed dim: 128
- Layers: 6
- Parameters: 1.3M
- File size: 5.2 MB

## Training
- Tasks: 25 diverse color/spatial variations
- Steps: 3000
- Method: TTT + Few-Shot

## Version History
- v1-v3: Internal development
- **v4**: First public release (64% training, 58% eval)

## Usage
```python
from src.cortex.model import CortexModel
import torch

# Load model
model = CortexModel(embed_dim=128, num_layers=6)
checkpoint = torch.load('models/cortex_v4.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict with few-shot
prediction = model.predict(
    test_input,
    example_inputs=[ex.input for ex in examples],
    example_outputs=[ex.output for ex in examples]
)
```

## Tasks Trained On (25 variations)

### Color Operations
- identity
- mask_smallest, mask_largest, mask_most_frequent, mask_least_frequent, mask_random
- fill_most_frequent, fill_least_frequent, fill_smallest, fill_largest
- swap_smallest_two, swap_largest_two, replace_smallest_with_largest, replace_largest_with_smallest
- increment_colors, decrement_colors

### Spatial Operations
- scale_2x, scale_3x, shrink_half
- flip_horizontal, flip_vertical
- rotate_90, rotate_180, rotate_270
- transpose
