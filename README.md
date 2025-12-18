# 🧠 Cortex-ARC

> A brain-inspired architecture for solving ARC-AGI puzzles (v1 & v2)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What is This?

**Cortex-ARC** is a modular, brain-inspired AI system designed to solve the [ARC-AGI](https://arcprize.org) benchmark — a test of general intelligence through abstract visual reasoning puzzles.

### Architecture

The system is organized like regions of the brain:

```
┌─────────────────────────────────────────────────────────────┐
│                      Cortex-ARC Brain                        │
├──────────────────┬──────────────────┬───────────────────────┤
│   Visual Cortex   │    Reasoning     │     Rule Engine       │
│  (Perception)     │   (Matching)     │   (Transformation)    │
├──────────────────┼──────────────────┼───────────────────────┤
│ • Object Detection│ • Object Matcher │ • Translation         │
│ • Background Det. │ • Comparison     │ • Rotation/Flip       │
│ • Color Encoding  │ • Signatures     │ • Recolor             │
└──────────────────┴──────────────────┴───────────────────────┘
```

## 📊 Current Results

| Test Suite | Accuracy | Notes |
|------------|----------|-------|
| Synthetic Tests | **100%** (9/9) | Translation, rotation, flip, recolor |
| ARC-AGI-1 Rotation Puzzles | **100%** (5/5) | Pure rotation/flip tasks |
| ARC-AGI-1 Overall | 2% | Only handles simple transformations so far |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/mondeep0123/Cortex-ARC.git
cd Cortex-ARC

# Install dependencies
pip install -e .

# Download ARC dataset
python scripts/download_data.py --version arc1

# Run evaluation
python scripts/evaluate_phase3.py --arc --data data/arc-agi-1/training --n 50
```

## 🏗️ Project Structure

```
Cortex-ARC/
├── src/
│   ├── brain/                    # Brain-inspired modules
│   │   ├── visual/               # Visual Cortex
│   │   │   ├── perception.py     # Object detection, background detection
│   │   │   ├── reasoning.py      # Object matching, transformation detection
│   │   │   ├── solver.py         # Phase 3 solver
│   │   │   └── color_encoder.py  # Color understanding
│   │   ├── prefrontal/           # Decision making (planned)
│   │   ├── temporal/             # Sequence processing (planned)
│   │   └── memory/               # Pattern memory (planned)
│   ├── core/                     # Core abstractions
│   │   ├── grid.py               # Grid representation
│   │   ├── task.py               # Task structure
│   │   └── primitives.py         # DSL primitives
│   └── data/                     # Data loading
├── scripts/
│   ├── download_data.py          # Download ARC datasets
│   └── evaluate_phase3.py        # Run evaluation
├── CEREBRUM.md                   # Architecture design document
└── configs/                      # Configuration files
```

## 🧪 What's Implemented (Phase 3)

### ✅ Working
- **Object Detection** — Connected components algorithm
- **Background Detection** — Border-based heuristic
- **Object Matching** — Hungarian algorithm for correspondence
- **Transformation Detection** — Rotation (90°, 180°, 270°), Flip (H/V), Translation, Recolor
- **Rule Extraction** — Find consistent rules across training examples
- **Rule Application** — Apply detected rules to test input

### ❌ Not Yet Implemented
- Pattern filling
- Object scaling/duplication
- Conditional rules
- Counting/arithmetic
- Shape completion
- ML-based pattern recognition

## 📖 Architecture Document

For the complete brain-inspired architecture design, see [CEREBRUM.md](CEREBRUM.md).

## 🔬 Research Directions

1. **Add More Transformations** — Scaling, pattern fill, conditional rules
2. **ML Micro-Models** — Train small neural networks for specific task types
3. **Hybrid Reasoning** — Combine neural perception with symbolic rule application
4. **ARC-AGI 2** — Tackle the harder 2025 benchmark

## 📚 References

- [ARC Prize Official](https://arcprize.org)
- [ARC-AGI Paper](https://arxiv.org/abs/1911.01547)
- [Kaggle Competition](https://kaggle.com/competitions/arc-prize-2025)

## 📝 License

MIT License

---

Built with 🧠 by [@mondeep0123](https://github.com/mondeep0123)
