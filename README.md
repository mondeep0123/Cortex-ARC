# 🧠 ARC-AGI Solver

> A research codebase for tackling the ARC-AGI benchmark (both ARC-AGI-1 and ARC-AGI-2)

## 🎯 Goal

Beat the current state-of-the-art on ARC-AGI benchmarks through novel approaches combining:
- Program synthesis
- Test-time training
- Refinement loops
- Neural-symbolic reasoning

## 📊 Current SOTA (December 2025)

| Benchmark | Best Score | Our Target |
|-----------|------------|------------|
| ARC-AGI-1 | ~85%+ | 90%+ |
| ARC-AGI-2 | 54.2% (GPT-5.2 Pro) | 60%+ |
| ARC-AGI-2 (Kaggle) | 24% (NVARC) | 30%+ |

## 🏗️ Project Structure

```
arc-agi-solver/
├── README.md
├── requirements.txt
├── setup.py
├── configs/                    # Configuration files
│   ├── base.yaml
│   ├── arc1.yaml
│   └── arc2.yaml
├── data/                       # Dataset storage
│   ├── arc-agi-1/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── test/
│   └── arc-agi-2/
│       ├── training/
│       ├── evaluation/
│       └── test/
├── src/                        # Source code
│   ├── __init__.py
│   ├── data/                   # Data loading & processing
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── augmentation.py
│   │   └── preprocessing.py
│   ├── core/                   # Core abstractions
│   │   ├── __init__.py
│   │   ├── grid.py             # Grid representation
│   │   ├── task.py             # Task abstraction
│   │   ├── primitives.py       # DSL primitives
│   │   └── transforms.py       # Grid transformations
│   ├── solvers/                # Solver implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Base solver interface
│   │   ├── brute_force.py      # Baseline brute force
│   │   ├── program_synthesis.py # Program synthesis
│   │   ├── neural/             # Neural approaches
│   │   │   ├── __init__.py
│   │   │   ├── trm.py          # Tiny Recursive Model
│   │   │   ├── diffusion.py    # Diffusion-based
│   │   │   └── transformer.py  # Transformer-based
│   │   ├── symbolic/           # Symbolic approaches
│   │   │   ├── __init__.py
│   │   │   ├── dsl.py          # Domain-specific language
│   │   │   └── search.py       # Program search
│   │   └── hybrid/             # Hybrid approaches
│   │       ├── __init__.py
│   │       ├── refinement.py   # Refinement loops
│   │       └── neurosymbolic.py
│   ├── evaluation/             # Evaluation & metrics
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── submission.py       # Kaggle submission
│   └── visualization/          # Visualization tools
│       ├── __init__.py
│       ├── grid_viz.py
│       ├── task_viz.py
│       └── analysis.py
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_experiments.ipynb
│   └── 03_analysis.ipynb
├── experiments/                # Experiment tracking
│   ├── logs/
│   └── checkpoints/
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_grid.py
│   ├── test_solvers.py
│   └── test_evaluation.py
└── scripts/                    # Utility scripts
    ├── download_data.py
    ├── train.py
    ├── evaluate.py
    └── submit.py
```

## 🚀 Quick Start

```bash
# Clone and setup
cd arc-agi-solver
pip install -e .

# Download datasets
python scripts/download_data.py --version both

# Run baseline evaluation
python scripts/evaluate.py --solver brute_force --dataset arc1

# Train a model
python scripts/train.py --config configs/arc2.yaml
```

## 📚 Key Concepts

### Grid Representation
- 2D arrays of integers (0-9 representing colors)
- Dimensions: 1x1 to 30x30
- Colors: 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey, 6=magenta, 7=orange, 8=cyan, 9=maroon

### Task Structure
```json
{
  "train": [
    {"input": [[...]], "output": [[...]]}
  ],
  "test": [
    {"input": [[...]], "output": [[...]]}
  ]
}
```

### Evaluation
- Exact match required
- 2 attempts per test case
- Final score = % of correct predictions

## 🔬 Research Directions

1. **Program Synthesis** - Generate executable programs from examples
2. **Test-Time Training** - Adapt models on the fly for each task
3. **Refinement Loops** - Iteratively improve predictions
4. **Neurosymbolic** - Combine neural perception with symbolic reasoning
5. **Compression-based** - Use information-theoretic approaches

## 📖 References

- [ARC Prize Official](https://arcprize.org)
- [ARC-AGI Paper](https://arxiv.org/abs/1911.01547)
- [Kaggle Competition](https://kaggle.com/competitions/arc-prize-2025)

## 📝 License

MIT License
