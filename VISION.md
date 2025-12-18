# 🎯 Cortex-ARC Vision

> One Model. All Abilities. Any Domain.

---

## The Core Insight

**The brain is not modular in the way software is modular.**

Brain "regions" are not separate programs. They're densely interconnected parts of ONE neural network. Abilities don't live in isolated modules — they EMERGE from the unified learning of the whole system.

Our model follows this principle:

```
NOT:  ColorModule + SpatialModule + PatternModule → Stitch together
YES:  ONE model → Train on diverse tasks → All abilities emerge
```

---

## What We're Building

### A Unified Reasoning Core

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│                    ONE SET OF WEIGHTS                            │
│                                                                  │
│   Learns through training:                                       │
│                                                                  │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│   │  Color  │  │ Spatial │  │ Pattern │  │ Objects │            │
│   │ Ability │  │ Ability │  │ Ability │  │ Ability │            │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│        │            │            │            │                  │
│        └────────────┴────────────┴────────────┘                  │
│                         │                                        │
│                         ▼                                        │
│              ┌─────────────────────┐                             │
│              │  Relations Ability  │                             │
│              └──────────┬──────────┘                             │
│                         │                                        │
│                         ▼                                        │
│              ┌─────────────────────┐                             │
│              │  Reasoning Ability  │                             │
│              └─────────────────────┘                             │
│                                                                  │
│   All abilities are EMERGENT from the same weights.              │
│   Not separate models. Not stitched together.                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## How Abilities Emerge

### Brain Analogy

```
Human brain:
  - ~86 billion neurons
  - ONE connected network
  - Different regions specialize through DEVELOPMENT and LEARNING
  - V4 "specializes" in color because it receives that input
  - Parietal "specializes" in space because of its connectivity
  - BUT they're all part of the SAME network
```

### Our Model

```
Cortex model:
  - ~10M parameters (small, efficient)
  - ONE connected network
  - Different abilities emerge through TRAINING
  - Color ability emerges from color-relevant patterns in data
  - Spatial ability emerges from position-relevant patterns
  - BUT they're all in the SAME weights
```

---

## Training Philosophy

### Not Curriculum of Separate Skills

```
❌ WRONG:
  1. Train Color Model
  2. Train Spatial Model
  3. Train Pattern Model
  4. Somehow combine them
```

### Unified Learning

```
✅ RIGHT:
  1. Train ONE model on ALL tasks
  2. Tasks naturally require multiple abilities
  3. Model learns to compose abilities automatically
  4. Abilities share representations
```

### Example: Learning from ARC

```
ARC Task: "Move the blue object right"

To solve, model must:
  • Understand "blue" (color ability)
  • Understand "object" (segmentation ability)
  • Understand "right" (spatial ability)
  • Understand "move" (transformation ability)

These abilities develop TOGETHER, not separately.
```

---

## Multi-Domain Generalization

### The Key: Preprocessing

```
Domain → Preprocessor → Grid → Model → Output

The MODEL is domain-agnostic.
Only PREPROCESSING is domain-specific.
```

### Examples

**Chess:**
```python
def chess_to_grid(board):
    # Convert 8x8 board to grid
    # Pieces become colors 1-6
    # Empty = 0
    return grid

# Training: (board_before, board_after) pairs
# Model learns piece movements
```

**Sudoku:**
```python
def sudoku_to_grid(puzzle):
    # 9x9 grid, numbers 0-9
    # Empty cells = 0
    return grid

# Training: (incomplete, complete) pairs
# Model learns constraint satisfaction
```

**Any New Game:**
```python
def new_game_to_grid(state):
    # Convert game state to grid
    return grid

# Just write the preprocessor!
# Model's abilities transfer
```

---

## Why One Model Works

### Shared Representations

```
"Blue object in top-left"

Color representation: [blue]
Spatial representation: [top-left]
Object representation: [contiguous region]

These representations are SHARED across all tasks.
Learning one task helps all other tasks.
```

### Compositionality

```
Task A: Learn "blue"
Task B: Learn "top-left"
Task C: Learn "move"

New Task: "Move blue object from top-left to bottom-right"
  → Compose existing abilities
  → No retraining needed
```

### Efficiency

```
Separate models:
  ColorModel: 5M params
  SpatialModel: 5M params
  PatternModel: 5M params
  ObjectModel: 5M params
  TOTAL: 20M params + communication overhead

Unified model:
  CortexModel: 10M params
  TOTAL: 10M params, naturally integrated
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CORTEX MODEL                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ INPUT ENCODING                                              │ │
│  │   Grid → Learned embeddings                                 │ │
│  │   Position encoding                                         │ │
│  │   Color encoding (learned, not hardcoded)                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ REASONING CORE                                              │ │
│  │   Transformer/Attention layers                              │ │
│  │   Learns all abilities in shared weights                    │ │
│  │   Recursive: Can refine answer over multiple passes         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ OUTPUT DECODING                                             │ │
│  │   Embeddings → Grid                                         │ │
│  │   Autoregressive or direct prediction                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Comparison

| Aspect | Modular (Separate) | Unified (Ours) |
|--------|-------------------|----------------|
| Architecture | Multiple models | One model |
| Communication | Explicit, complex | Implicit, natural |
| Training | Separate, then combine | End-to-end |
| Representations | Separate per module | Shared |
| Compositionality | Hard | Natural |
| Proven | ❌ No winner | ✅ TRM shows it works |

---

## Roadmap

### Phase 1: Core Model
- Design unified architecture
- Implement training loop
- Train on ARC-AGI
- Validate: 40%+ ARC-AGI-1

### Phase 2: Multi-Domain
- Chess, Sudoku, Minesweeper preprocessors
- Validate transfer learning
- Fine-tune if needed

### Phase 3: Multi-Modal
- Image encoder → Unified space
- Text encoder → Unified space
- Same reasoning core

---

## Success Criteria

```
1. ONE model handles ALL ARC tasks
2. Same model transfers to Chess/Sudoku via preprocessing
3. Abilities compose (novel combinations work)
4. Generalizes to ARC-AGI-2 (never seen during training)
```

---

## Summary

We're not building a committee of experts.
We're not stitching modules together.
We're building ONE brain that learns to reason.

Different abilities emerge naturally from unified training.
That's how real brains work.
That's how our model works.
