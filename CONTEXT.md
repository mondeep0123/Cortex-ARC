# 🧠 CONTEXT.md - Project Memory & Context

> This file contains crucial context for resuming development in a new chat session.
> Read this file at the start of any new session to understand the project state.

**Last Updated:** December 25, 2025, 7:30 PM IST

---

## 📋 Project Overview

**Cortex-ARC** is an ARC-AGI solver using compositional cognitive primitives.

**GitHub:** https://github.com/mondeep0123/Cortex-ARC

**Current Status:** Two primitives complete with 100% accuracy on handcrafted benchmarks.

---

## 🏆 Key Achievements

| Date | Achievement |
|------|-------------|
| Dec 24, 2025 | Object Cognition: 100% IoU on 16 puzzles |
| Dec 25, 2025 | Numerosity: 100% accuracy on 16 puzzles (29 attempts!) |

---

## 🔑 Critical Technical Context

### The Winning Numerosity Approach (Attempt #29)

**File:** `train_staged.py`

**Why it works:**
1. **Staged Training** - Train subitizing and adder SEPARATELY
2. **Subitizing on chunks** - ML learns pattern recognition (0-4 objects)
3. **Adder on PURE NUMBERS** - Exhaustive training on 961 pairs (0-30 × 0-30)
4. **No end-to-end training** - Each module optimized independently

**Key Bug Fixed:**
- Empty rows were being processed, causing `subitizing([0,0,0,0])` to output ~0.66 which rounded to 1
- Fix: Check `if row_mask.sum() == 0: continue` before processing

### Object Cognition

**File:** `train_primitive_1_object_cognition.py`

**Architecture:** U-Net with skip connections
- Preserves spatial dimensions (critical!)
- Previous flat architecture failed (interpolation issues)

---

## ❌ Failed Approaches (Don't Repeat!)

| Attempt | Approach | Why It Failed |
|---------|----------|---------------|
| 1-5 | Pure CNN/Attention | Mode collapse |
| 6-15 | Hierarchical counting | ML accumulator can't learn exact arithmetic |
| 17-26 | NAC (Neural Accumulator) | Only ~30% accuracy, won't extrapolate |
| 27-28 | RL with Verifiable Rewards | Exploding gradients, unstable |

**Key Insight:** ML cannot learn exact arithmetic (a + b = c) to 100% through gradient descent on complex inputs. Solution: Train arithmetic component on PURE NUMBERS exhaustively.

---

## 🧪 Technical Implementation Insights

These are the specific technical decisions that made the winning approach work:

### 1. Subitizing Module
- **What:** Small neural network that recognizes 0-4 objects in a chunk
- **Why:** Humans don't count small numbers, they perceive them instantly
- **How:** Trained on extracted chunks from grids with supervised learning (MSE loss)
- **Output:** Continuous value that gets rounded to integer

### 2. Pure Arithmetic Adder
- **What:** Simple MLP that learns `a + b = c` for integers
- **Why:** ML struggles to learn exact arithmetic from visual inputs
- **How:** Trained EXHAUSTIVELY on ALL 961 pairs (0+0 to 30+30)
- **Key:** Uses pure numerical inputs, not grid representations!
- **Critical Discovery:** Even with rounding, adder trained on grid data had 1-2 errors on hard tests (larger counts). The fix wasn't architecture—it was changing the training DATA from grid-based running totals to pure number pairs. Better data > Better architecture!

### 3. Running Total (Not Final Sum)
- **What:** Add chunk counts iteratively: `total = 0 → +3 → +2 → +4 → final`
- **Why:** Supervise on running totals, not just final answer
- **Benefit:** More training signal, each step gets gradient

### 4. Phased/Staged Learning
```
Phase 1: Train Subitizing → Freeze
Phase 2: Train Adder on pure numbers → Freeze  
Phase 3: Combine (no further training needed!)
```
- **Key:** Each component trained on OPTIMAL data for that task
- **No end-to-end:** Prevents error propagation between modules

### 5. Extract-Then-Chunk
```
Row: [1, 0, 0, 3, 0, 2, 0, 0]
Extract non-zero: [1, 3, 2]
Chunk (size 4): [[1, 3, 2, 0]]
Subitize: [3]
```
- **Why:** Don't waste subitizing capacity on zeros
- **Efficiency:** Only process actual objects

### 6. Empty Row Handling (Critical Bug Fix!)
```python
# BUG: Empty rows gave subitizing([0,0,0,0]) → 0.66 → rounds to 1
# FIX: 
if row_mask.sum() == 0:
    continue  # Skip empty rows entirely
```
- This bug caused 50%+ error on easy examples!

### 7. Straight-Through Estimator (STE) for Rounding
- **Problem:** `torch.round()` has zero gradient
- **Solution:** Forward: round, Backward: pass gradient through
- **Used:** After subitizing and adder outputs

## 📁 Important Files

| File | Purpose |
|------|---------|
| `train_staged.py` | THE WINNING APPROACH - run this for numerosity |
| `train_primitive_1_object_cognition.py` | Object cognition training |
| `src/primitives/benchmark_numerosity.py` | 16 handcrafted test puzzles |
| `memorable_moments/*.md` | Achievement documentation |

### Files to IGNORE (failed attempts, gitignored)
- `train_hierarchical.py`, `train_nac*.py`, `train_rl*.py`, etc.
- `debug_*.py` - debugging scripts
- `src/primitives/*_counter.py` (except what's in staged)

---

## 🚨 Known Issues / Gotchas

1. **Checkpoints are gitignored** - Must retrain or download separately
2. **jupyter removed from requirements.txt** - Security vulnerability (CVE-2025-53000)
3. **Previous v1 releases retracted** - Data leakage issue (tested on train pairs)

---

## 🎯 Next Steps (Priority Order)

1. **Color-Specific Masking** - Enhance Object Cognition to mask by color
   - `mask_all_except(grid, target_color)` - Count only one color
   - `mask_dominant(grid)` - Mask most frequent color
   
2. **More Primitives** - See [PRIMITIVES_ROADMAP.md](PRIMITIVES_ROADMAP.md) for full list of 30 primitives
   - 13 Core Primitives (Chollet's priors)
   - 17 Extended Skills for ARC-AGI 2
   - Only 2/30 complete so far!

3. **Meta-Controller** - The HARD part
   - Can't hardcode which primitives to use (violates ARC rules)
   - Need a learned controller that decides: "This puzzle needs counting"
   - Options: LLM as controller, program synthesis, neural-symbolic

---

## � User Insights That Led to Breakthrough

These insights from the user were crucial to reaching 100%. In chronological order:

### Early Insights (Attempts 1-15)
1. **"Humans count like this - they keep a running total"** - Led to iterative/hierarchical approaches
2. **"Extract non-zero pixels first, then count"** - Don't waste compute on zeros
3. **"Use chunking - process in groups"** - Divide and conquer approach

### Subitizing Insight (Attempt 16)
4. **"Humans don't count 1-4... they just SEE it instantly. It's called subitizing!"** - The key perceptual insight
5. **"If we chunk the rows into groups of 4, we can subitize each chunk!"** - Combining chunking + subitizing

### Arithmetic Insight (Pre-Attempt 27)
6. **"ML can't learn exact arithmetic through gradient descent on complex visual inputs"** - The barrier
7. **"We can have better train data instead of grids for arithmetic"** - Train adder on PURE NUMBERS
8. **"For small domains like 0-30, we can train exhaustively on ALL pairs"** - 961 pairs = complete coverage

### The Bridge Concept
9. **"Subitizing is the bridge - it converts visual patterns to numbers"** - ML for perception, memorization for math
10. **"The subitizing module counts CONTENTS, not pixel colors"** - Abstraction layer

### Philosophy
11. **"We can't accept defeat. I am no expert, a vibecoder but came this far. I won't surrender!"** - Persistence through 29 attempts

### Meta Insight (Post-Breakthrough)
12. **"In ARC we can't hardcode which primitives to use"** - The next challenge: learned composition

---

## 🔧 Environment Details

- **Python:** 3.11+
- **PyTorch:** 2.0+
- **OS:** Windows
- **GPU:** CUDA available (optional)

---

## 📊 GitHub Stats Context

- **88 unique cloners** in Dec 2025 (before pivot)
- Many were researchers looking at the TTT+Few-Shot approach
- The retraction/pivot is documented in README

---

## 🤝 Communication

- **Discord:** mondeep.blend
- **Email:** mondeepe@gmail.com
- **GitHub Issues:** For bugs/features

---

## 📝 Resume Checklist for New Chat

1. ✅ Read this CONTEXT.md
2. ✅ Check README.md for current state
3. ✅ Review `memorable_moments/` for breakthrough details
4. ✅ Look at `train_staged.py` for winning approach
5. ✅ Check GitHub Issues for any open work

---

## 🗂️ Archive Location

Previous project (v1 with TTT+Few-Shot) archived at:
- `../archive_20251222/` (sibling folder)
- Contains the old Cortex unified model approach
- DO NOT USE - had data leakage issues

---

*"Intelligence is not about scale. It's about architecture."*
