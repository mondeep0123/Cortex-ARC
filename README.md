# 🧠 Cortex-ARC

> A brain-inspired **learning architecture** for general reasoning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ⚠️ Vision Clarification

**This project is NOT about hardcoding patterns.**

The goal is to build a system that **LEARNS fundamental cognitive abilities** — not puzzle-specific classifiers.

---

## 🎯 True Vision

### Core Principle: Train Cognitive Experts, Not Puzzle Solvers

```
❌ WRONG: Train "RotationDetector", "CropDetector", "FlipDetector"
           → These are just classifiers for specific puzzles
           
✅ RIGHT: Train "ColorExpert", "SpatialExpert", "PatternExpert"
           → These are fundamental cognitive abilities
           → They COMPOSE to solve ANY puzzle
```

### The Micro-Model Philosophy

Each micro-model is an **expert in a cognitive domain**, not a puzzle type:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE MICRO-MODELS                           │
│                                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│   │    Color     │  │   Spatial    │  │   Pattern    │             │
│   │   Expert     │  │   Expert     │  │   Expert     │             │
│   │              │  │              │  │              │             │
│   │ Understands: │  │ Understands: │  │ Understands: │             │
│   │ • Hue        │  │ • Position   │  │ • Repetition │             │
│   │ • Contrast   │  │ • Distance   │  │ • Symmetry   │             │
│   │ • Grouping   │  │ • Direction  │  │ • Sequence   │             │
│   │ • Similarity │  │ • Boundaries │  │ • Hierarchy  │             │
│   └──────────────┘  └──────────────┘  └──────────────┘             │
│          │                 │                 │                      │
│          └─────────────────┼─────────────────┘                      │
│                            ▼                                        │
│                  ┌──────────────────┐                               │
│                  │   Object Expert   │                              │
│                  │                   │                              │
│                  │ Combines color,   │                              │
│                  │ spatial, pattern  │                              │
│                  │ to understand     │                              │
│                  │ OBJECTS           │                              │
│                  └──────────────────┘                               │
│                            │                                        │
│                            ▼                                        │
│                  ┌──────────────────┐                               │
│                  │  Relation Expert  │                              │
│                  │                   │                              │
│                  │ Understands how   │                              │
│                  │ objects RELATE    │                              │
│                  │ to each other     │                              │
│                  └──────────────────┘                               │
│                            │                                        │
│                            ▼                                        │
│                  ┌──────────────────┐                               │
│                  │ Reasoning Expert  │                              │
│                  │                   │                              │
│                  │ Uses all experts  │                              │
│                  │ to INFER rules    │                              │
│                  │ and apply them    │                              │
│                  └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘

Each expert is a LEARNED neural network.
Experts COMMUNICATE through shared representations.
Experts COMPOSE to solve any problem.
```

---

## 🧠 Why This Matters

### Puzzle-Specific vs Cognitive Abilities

| Puzzle-Specific (WRONG) | Cognitive (RIGHT) |
|-------------------------|-------------------|
| Detects rotation | Understands spatial relationships |
| Detects cropping | Understands boundaries and regions |
| Detects color swap | Understands color relationships |
| **Fails on new puzzles** | **Composes to solve new puzzles** |

### Example: Solving a "Rotation" Puzzle

**With Puzzle-Specific Approach:**
```
1. Hardcoded "RotationDetector" recognizes rotation
2. Apply np.rot90()
3. Done (but fails on ANY variation)
```

**With Cognitive Approach:**
```
1. Spatial Expert: "The pixel positions changed in a circular pattern"
2. Pattern Expert: "This matches the concept of angular transformation"
3. Relation Expert: "Input corners map to output corners with 90° shift"
4. Reasoning Expert: "Apply the same spatial transformation"
5. Works on ANY spatial transformation, not just hardcoded ones
```

---

## 🏗️ Cognitive Micro-Models

| Expert | Learns | Used For |
|--------|--------|----------|
| **Color Expert** | Color relationships, grouping, contrast | Understanding which pixels belong together |
| **Spatial Expert** | Positions, distances, directions | Understanding where things are |
| **Pattern Expert** | Repetition, symmetry, sequences | Finding structure in data |
| **Object Expert** | Entity boundaries, properties | Recognizing distinct objects |
| **Relation Expert** | How entities relate | Understanding transformations |
| **Memory Expert** | Store and retrieve | Learning from examples |
| **Reasoning Expert** | Inference, composition | Solving the puzzle |

---

## 📊 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Infrastructure | ✅ Complete | - |
| Hardcoded Baseline | ⚠️ Deprecated | Wrong approach, kept for reference |
| Color Expert | 📋 Planned | First cognitive micro-model |
| Spatial Expert | 📋 Planned | - |
| Pattern Expert | 📋 Planned | - |
| Orchestration | 📋 Planned | Communication between experts |

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

- [VISION.md](VISION.md) - Core philosophy: Learning cognitive abilities
- [CEREBRUM.md](CEREBRUM.md) - Full architecture design

---

## 🎯 The Goal

Build a system where:

1. **Input** can be text, grids, code, images — anything
2. **Cognitive experts** understand the fundamental structure
3. **Experts communicate** to form understanding
4. **Reasoning emerges** from composition
5. **Generalization** is automatic because we learned ABILITIES, not PATTERNS

---

*"Intelligence is not about knowing the answers. It's about knowing how to think."*
