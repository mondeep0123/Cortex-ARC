# 🧠 Cortex-ARC

> A unified learning architecture for general reasoning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Vision

**One model. All cognitive abilities. Any domain.**

We're building a unified neural network that learns fundamental reasoning abilities — color understanding, spatial awareness, pattern recognition, relational thinking — all within the **same set of weights**.

Not separate modules. Not hardcoded rules. One brain.

---

## 🧠 Core Principles

### 1. ONE Unified Model
```
The brain is one network, not separate organs.
Our model is one network, not stitched modules.
Abilities EMERGE from training, not from separate architectures.
```

### 2. Learn ALL Abilities Together
```
Color + Spatial + Pattern + Objects + Relations + Reasoning
                        ↓
              SAME weights learn ALL
                        ↓
           Abilities naturally compose
```

### 3. Multi-Domain via Preprocessing
```
Chess    → Preprocess to Grid → Model → Move
Sudoku   → Preprocess to Grid → Model → Solution
ARC-AGI  → Already Grid       → Model → Answer
New Game → Write preprocessor → Model → Works
```

### 4. Multi-Modal (Future)
```
Phase 1: Grids (now)
Phase 2: Images → Grid-like encoding → Model
Phase 3: Text → Token encoding → Model
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CORTEX UNIFIED MODEL                            │
│                                                                      │
│   Input: Grid (or encoded input from any domain)                     │
│                              │                                       │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                     ENCODER                                   │  │
│   │   Embeds input into learned representation space              │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                  REASONING CORE                               │  │
│   │                                                               │  │
│   │   Learns through training:                                    │  │
│   │   • Color relationships                                       │  │
│   │   • Spatial relationships                                     │  │
│   │   • Pattern recognition                                       │  │
│   │   • Object understanding                                      │  │
│   │   • Relational reasoning                                      │  │
│   │                                                               │  │
│   │   All abilities in SHARED WEIGHTS                             │  │
│   │                                                               │  │
│   │   Recursive: Refines answer iteratively                       │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                     DECODER                                   │  │
│   │   Generates output (grid, move, answer)                       │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│   Output: Predicted grid/answer                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Brain Inspiration

The model learns abilities that correspond to brain functions:

| Brain Region | Ability | How It's Learned |
|--------------|---------|------------------|
| V4 | Color understanding | Same weights |
| Parietal | Spatial reasoning | Same weights |
| Temporal | Pattern recognition | Same weights |
| Fusiform | Object detection | Same weights |
| Angular Gyrus | Relations | Same weights |
| Prefrontal | Reasoning | Same weights |

**Not separate models — abilities EMERGE in a unified network through training.**

---

## 📊 Roadmap

### Phase 1: Grid Reasoning (Current)
- [ ] Design unified architecture
- [ ] Train on ARC-AGI tasks
- [ ] Target: 40%+ on ARC-AGI-1
- [ ] Test transfer: Chess, Sudoku, Minesweeper via preprocessing

### Phase 2: Multi-Modal
- [ ] Add image encoder (vision)
- [ ] Add text encoder (NLP)
- [ ] Unified representation space

### Phase 3: General Reasoning
- [ ] Natural language I/O
- [ ] Explain reasoning
- [ ] Novel domain generalization

---

## 🚀 Quick Start

```bash
git clone https://github.com/mondeep0123/Cortex-ARC.git
cd Cortex-ARC
pip install -e .
python scripts/download_data.py --version arc1
```

---

## 📚 Documentation

- [VISION.md](VISION.md) - Core philosophy
- [CEREBRUM.md](CEREBRUM.md) - Full architecture design

---

## 🎯 Why This Approach?

| Other Approaches | Our Approach |
|-----------------|--------------|
| Hardcoded rules | Learned abilities |
| Separate modules | Unified model |
| Domain-specific | Domain-agnostic (via preprocessing) |
| Scale = intelligence | Architecture = intelligence |

---

## 📈 Target Performance

| Benchmark | Target | Notes |
|-----------|--------|-------|
| ARC-AGI-1 | 40%+ | Primary benchmark |
| ARC-AGI-2 | 25%+ | Generalization test |
| Chess | Works | Via preprocessing |
| Sudoku | Works | Via preprocessing |
| New domains | Works | Just add preprocessor |

---

*"Intelligence is not about scale. It's about architecture."*
