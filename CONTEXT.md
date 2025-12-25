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
   
2. **More Primitives**
   - Geometry (shapes, lines, regions)
   - Topology (connectivity, holes)
   - Symmetry detection

3. **Meta-Controller** - The HARD part
   - Can't hardcode which primitives to use (violates ARC rules)
   - Need a learned controller that decides: "This puzzle needs counting"
   - Options: LLM as controller, program synthesis, neural-symbolic

---

## 💬 User Quotes (Philosophy)

> "We can't accept defeat. I am no expert, a vibecoder but came this far. I won't surrender!"

> "Humans don't count 1-4... they just SEE it instantly. It's called subitizing!"

> "We can have better train data instead of grids for arithmetic"

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
