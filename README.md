# 🧠 Cortex-ARC: Numerosity & Object Cognition

> Compositional primitives for ARC reasoning — 100% accuracy achieved!

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> ⚠️ **Notice regarding Cortex-ARC v1 (The TTT Solver)**
> 
> The project architecture has shifted from an end-to-end solver to a **"Cognitive Primitives"** system.
> 
> **Retraction:** The previous v1.0.0 releases have been **removed**. Post-release auditing revealed that the reported 60% accuracy was a result of **data leakage** (evaluation logic inadvertently accessed training targets). We have retracted these results to maintain scientific integrity.
> 
> **The Path Forward:** To prevent similar opacity issues, the new architecture focuses on **verified, compositional primitives** (e.g., Object Cognition, Numerosity) where accuracy is deterministic and independently testable.

---

## 🎯 Current Results

| Primitive | Benchmark | Accuracy | Method |
|-----------|-----------|----------|--------|
| Object Cognition | 16 Handcrafted | **100% IoU** | U-Net Spatial Preservation |
| Numerosity | 16 Handcrafted | **100%** | Staged Training (Subitizing + Arithmetic) |

Achieved with only **23K parameters** for Numerosity and **~1.2M** for Object Cognition.

> 🎄 **Christmas Day Breakthrough**: 100% Numerosity achieved on December 25, 2025 after 29 attempts!

<details>
<summary><b>📊 Numerosity Benchmark Results</b></summary>

```
Testing on 16 Handcrafted Puzzles:

empty               : Target= 0, Pred= 0 ✓
single              : Target= 1, Pred= 1 ✓
small_count         : Target= 5, Pred= 5 ✓
exact_ten           : Target=10, Pred=10 ✓
single_color_many   : Target= 7, Pred= 7 ✓
two_colors          : Target= 7, Pred= 7 ✓
dominant_color      : Target=12, Pred=12 ✓
equal_colors        : Target= 6, Pred= 6 ✓
many_colors         : Target= 9, Pred= 9 ✓
sparse              : Target= 2, Pred= 2 ✓
dense               : Target=10, Pred=10 ✓
checkerboard        : Target= 8, Pred= 8 ✓
large_count         : Target=21, Pred=21 ✓
rare_color          : Target=18, Pred=18 ✓
close_tie           : Target= 9, Pred= 9 ✓
full_grid           : Target= 9, Pred= 9 ✓

RESULTS: 16/16 = 100.0%
```

</details>

<details>
<summary><b>📊 Object Cognition Results</b></summary>

```
Object Cognition Benchmark:

Training Set:   100.00% IoU
Validation Set: 100.00% IoU
Handcrafted:    100.00% IoU (16/16 perfect)

Verdict: EXCELLENT - Model is ARC-ready!
```

</details>

---

## 🚀 The Journey: 29 Attempts to Perfection

```
Attempt 1-5:   ~20%  (Mode collapse, CNN/Attention failures)
Attempt 6-10:  ~35%  (Learning something, but unstable)
Attempt 11-15: ~50%  (Hierarchical approaches)
Attempt 16:    ~75%  (Ultimate Counter - previous peak!)
Attempt 17-26: ~30%  (Binary adder struggles - NAC, RL, etc.)
Attempt 27-28: ~25%  (RL with Verifiable Rewards - exploding)
Attempt 29:    100%  (STAGED TRAINING!) 🎉
```

<details>
<summary><b>📋 Key User Insights That Led to 100%</b></summary>

> "It's a running total! Humans don't add ALL row counts at once - they keep a running count!"

> "Humans don't count 1-4... they just SEE it instantly. It's called subitizing!"

> "If we chunk the rows into groups of 4, we can subitize each chunk!"

> "We can have better train data instead of grids for arithmetic"

> "We can't accept defeat. I am no expert, a vibecoder but came this far. I won't surrender!"

</details>

---

## 🎯 Vision: Compositional Primitives

**Break complex reasoning into learnable skills. Combine them to solve any problem.**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPOSITIONAL PIPELINE                            │
│                                                                      │
│   Input Grid                                                         │
│       ↓                                                             │
│   ┌──────────────────┐                                              │
│   │ OBJECT COGNITION │  ← Primitive #1: What's in the grid?        │
│   │    (100% IoU)    │     Segments foreground from background      │
│   └──────────────────┘                                              │
│       ↓                                                             │
│   ┌──────────────────┐                                              │
│   │   NUMEROSITY     │  ← Primitive #2: How many objects?          │
│   │    (100% Acc)    │     Counts using subitizing + addition       │
│   └──────────────────┘                                              │
│       ↓                                                             │
│   ┌──────────────────┐                                              │
│   │   (FUTURE)       │  ← More primitives coming...                │
│   │ Geometry, Topo   │     Spatial relationships, patterns, etc.   │
│   └──────────────────┘                                              │
│       ↓                                                             │
│   Output: Reasoning Result                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture: The Winning Formula

### Numerosity Counter (23K params)

```
Grid → Extract Non-Zero → Chunks of 4 → Subitize → Add → Total
                              ↓              ↓
                          Variable       ML Pattern    Memorized
                          Length         Recognition   Lookup
                                          (100%)       (100%)
```

| Component | Training Data | Method | Accuracy |
|-----------|---------------|--------|----------|
| **Subitizing** | Chunks from grids | Supervised (MSE) | 100% |
| **Adder** | ALL 961 pairs (0-30)×(0-30) | Supervised (MSE) | 100% |

### Object Cognition (1.2M params)

```
Grid → Color Embedding → U-Net Encoder → Bottleneck → U-Net Decoder → Mask
                         (Skip connections preserve spatial info!)
```

---

## 🧠 Key Techniques

### 1. Staged Training (The Breakthrough!)
```python
# Stage 1: Train subitizing on chunks (pattern recognition)
for chunk in all_chunks:
    loss = (subitizing(chunk) - true_count) ** 2
    
# Stage 2: Train adder on PURE NUMBERS (exhaustive!)
for a in range(31):
    for b in range(31):
        loss = (adder(a, b) - (a + b)) ** 2
        
# Stage 3: Combine - NO fine-tuning needed!
total = subitizing → adder → result  # 100%!
```

### 2. Subitizing (Human-like Perception)
```python
# Humans don't count 0-4, they SEE it instantly
chunk = [1, 0, 1, 1]  # 3 objects
count = subitizing(chunk)  # Output: 3 (instant recognition!)
```

### 3. Memorized Arithmetic
```python
# For small domains, memorization = perfect accuracy
# 961 pairs = complete lookup table for 0-30 addition
adder.train_on(all_pairs)  # 100% accuracy guaranteed!
```

<details>
<summary><b>🔬 Deep Dive: All Technical Implementation Insights</b></summary>

### 4. Running Total (Not Final Sum)
Instead of supervising only the final answer, we supervise each step:
```python
# Each addition step gets gradient signal
total = 0
total = adder(total, chunk1_count)  # Supervise here
total = adder(total, chunk2_count)  # And here
total = adder(total, chunk3_count)  # And here
# More training signal = better learning
```

### 5. Extract-Then-Chunk Pipeline
```
Row: [1, 0, 0, 3, 0, 2, 0, 0]
     ↓ Extract non-zero
[1, 3, 2]
     ↓ Pad to chunk size (4)
[1, 3, 2, 0]
     ↓ Subitize
Count: 3
```
- **Why:** Don't waste subitizing capacity on zeros
- **Benefit:** Denser signal for pattern recognition

### 6. Phased/Staged Learning (Critical!)
```
┌─────────────────────────────────────────┐
│ Phase 1: Subitizing Training            │
│   Data: Chunks extracted from grids     │
│   Target: Count of non-zero in chunk    │
│   Result: 100% on 0-4 recognition       │
│   → FREEZE weights                      │
├─────────────────────────────────────────┤
│ Phase 2: Adder Training                 │
│   Data: ALL 961 pairs (0-30 × 0-30)     │
│   Target: a + b                         │
│   Result: 100% on addition              │
│   → FREEZE weights                      │
├─────────────────────────────────────────┤
│ Phase 3: Combined Inference             │
│   No training needed!                   │
│   Just connect: Subitizing → Adder      │
│   Result: 100% end-to-end               │
└─────────────────────────────────────────┘
```

### 7. The Critical Bug Fix (Empty Rows)
```python
# BUG: Empty rows still processed
# subitizing([0, 0, 0, 0]) → outputs ~0.66 → rounds to 1
# Result: "Ghost counts" for empty rows!

# Example bug impact:
# Grid with 4 objects + 4 empty rows
# Predicted: 4 (real) + 4 (ghost) = 8 ❌
# Expected: 4 ✓

# FIX: Skip empty rows entirely
for row in grid:
    row_mask = (row > 0).float()
    if row_mask.sum() == 0:
        continue  # Skip! No objects here
```
This single bug caused **50%+ error** on easy examples!

### 8. Straight-Through Estimator (STE)
```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)  # Round in forward
    
    @staticmethod
    def backward(ctx, grad):
        return grad  # Pass gradient through unchanged

# Used after subitizing: count = RoundSTE.apply(raw_count)
# Enables gradient flow through rounding operation
```

### 9. Why End-to-End Training Failed
```
End-to-End:
  Visual Input → [Subitizing → Adder] → Loss
                        ↑
              Gradients corrupted by
              compounding errors!

Staged:
  Chunks → Subitizing → Loss ✓ (optimal for perception)
  Numbers → Adder → Loss ✓ (optimal for arithmetic)
  Combined → Just works! ✓
```

</details>

---

## 🚀 Quick Start

### Step 1: Clone & Install
```bash
git clone https://github.com/mondeep0123/Cortex-ARC.git
cd Cortex-ARC
pip install -r requirements.txt
```

### Step 2: Train Numerosity (5 minutes)
```bash
python train_staged.py
```

### Step 3: Test on Benchmark
```python
from train_staged import StagedCounter
import torch

model = StagedCounter()
model.load_state_dict(torch.load('checkpoints/staged_model.pt')['model_state'])
model.eval()

# Count objects in any grid!
grid = torch.tensor([[1, 0, 2, 0], [0, 3, 0, 4]])
mask = (grid > 0).float()
count = model(grid.unsqueeze(0), mask.unsqueeze(0))
print(f"Objects: {int(count.item())}")  # Output: 4
```

---

## 📊 Roadmap

### ✅ Phase 1: Core Primitives (COMPLETE!)
- [x] Object Cognition - 100% IoU
- [x] Numerosity - 100% Accuracy
- [x] Compositional integration
- [x] Handcrafted benchmarks

### 🔄 Phase 2: Enhanced Primitives (IN PROGRESS)
- [ ] Color-specific masking ("Count red pixels")
- [ ] Dominant/Rare color detection
- [ ] Object instance segmentation

### ⏳ Phase 3: Spatial Primitives
- [ ] Geometry (shapes, lines, regions)
- [ ] Topology (connectivity, holes)
- [ ] Symmetry detection

### ⏳ Phase 4: Integration
- [ ] Multi-primitive reasoning
- [ ] ARC-AGI full benchmark
- [ ] Unified solver

---

## 📁 Project Structure

```
Cortex-ARC/
├── src/primitives/
│   ├── object_cognition_primitive.py  # U-Net segmentation
│   ├── benchmark_numerosity.py        # 16 handcrafted puzzles
│   ├── distillation_counting.py       # Pattern generator
│   └── ...
├── train_staged.py           # THE WINNING APPROACH! 🏆
├── train_primitive_1_object_cognition.py
├── memorable_moments/        # Achievement documentation
│   ├── numerosity_breakthrough_100_percent.md
│   └── 2025-12-24_object_cognition_breakthrough.md
├── checkpoints/              # Saved models (gitignored)
└── configs/
```

---

## 🎓 Lessons Learned

| Insight | Implementation |
|---------|----------------|
| **ML for patterns, algorithms for math** | "Algorithms" can be learned via memorization! |
| **Staged training works** | Train each component on optimal data |
| **Exhaustive > Curriculum** | For small domains, cover EVERYTHING |
| **Debugging is crucial** | Empty row bug cost 50% accuracy! |
| **Human intuition + AI** | User insights led to breakthrough |
| **Never surrender** | 29 attempts to reach 100% |

---

## 📈 Model Comparison

| Model | Params | Benchmark | Accuracy |
|-------|--------|-----------|----------|
| Numerosity (Staged) | **23K** | Handcrafted 16 | **100%** |
| Object Cognition | 1.2M | Handcrafted 16 | **100% IoU** |
| Ultimate Counter (v16) | ~30K | Generated | 75% |
| NAC Approaches | ~6K | Generated | ~30% |
| RL Approaches | ~23K | Generated | ~25% |

---

## 📚 Documentation

- **[Numerosity Breakthrough](memorable_moments/numerosity_breakthrough_100_percent.md)** - Full journey documenting 29 attempts
- **[Object Cognition Breakthrough](memorable_moments/2025-12-24_object_cognition_breakthrough.md)** - U-Net success story

---

## 📞 Contact

For questions, collaborations, or feedback:

- **Discord**: [mondeep.blend](https://discord.com/users/1085083654251872357)
- **GitHub Issues**: [Open an issue](https://github.com/mondeep0123/Cortex-ARC/issues)

---

## 🌟 Contributing

Contributions welcome! Areas to work on:
- Color-specific masking
- Spatial primitives (geometry, topology)
- More benchmark puzzles
- Integration with reasoning modules

---

*"We can't accept defeat. I am no expert, a vibecoder but came this far. I won't surrender!"*

*— The User, Christmas Day 2025, 3:14 AM IST*

# 🏆 100% ACCURACY ACHIEVED 🏆
# 🎄 Merry Christmas! 🎄
