# Project Architecture

## Overview
This document outlines the technical architecture for the ARC-AGI curriculum learning system.

## Design Principles

1. **Modularity**: Each curriculum skill is a separate, testable module
2. **Composability**: Skills can be combined to solve complex tasks
3. **Interpretability**: Clear reasoning traces showing which skills are applied
4. **Scalability**: Easy to add new skills to the curriculum
5. **Measurability**: Each skill has quantifiable success metrics

---

## System Architecture

```
arc-agi-solver/
│
├── data/                          # ARC dataset
│   ├── training/                  # 400 training tasks
│   ├── evaluation/                # 400 evaluation tasks  
│   └── curriculum/                # Synthetic curriculum tasks
│       ├── object_cognition/
│       ├── numerosity/
│       ├── geometry/
│       └── ...
│
├── src/
│   ├── core/                      # Core knowledge priors
│   │   ├── object_cognition.py
│   │   ├── numerosity.py
│   │   ├── geometry.py
│   │   ├── topology.py
│   │   └── physics.py
│   │
│   ├── operations/                # Cognitive operations
│   │   ├── pattern_recognition.py
│   │   ├── transformation.py
│   │   ├── analogy.py
│   │   ├── goal_reasoning.py
│   │   └── hypothesis_testing.py
│   │
│   ├── meta/                      # Meta-cognitive skills
│   │   ├── attention.py
│   │   ├── working_memory.py
│   │   └── search.py
│   │
│   ├── curriculum/                # Curriculum training
│   │   ├── task_generator.py     # Generate synthetic tasks
│   │   ├── trainer.py             # Training loop
│   │   ├── scheduler.py           # Curriculum scheduling
│   │   └── evaluator.py           # Skill assessment
│   │
│   ├── model/                     # Neural architecture
│   │   ├── encoder.py             # Grid encoder
│   │   ├── reasoning.py           # Reasoning module
│   │   ├── decoder.py             # Grid decoder
│   │   └── skill_modules.py       # Skill-specific modules
│   │
│   ├── solver/                    # Task solver
│   │   ├── arc_solver.py          # Main solver
│   │   ├── skill_composer.py      # Combine skills
│   │   └── search_strategy.py     # Solution search
│   │
│   └── utils/                     # Utilities
│       ├── grid_utils.py          # Grid operations
│       ├── visualization.py       # Visualize tasks/solutions
│       └── metrics.py             # Evaluation metrics
│
├── tests/                         # Unit tests
│   ├── test_core/
│   ├── test_operations/
│   ├── test_meta/
│   └── test_curriculum/
│
├── configs/                       # Configuration files
│   ├── curriculum.yaml            # Curriculum schedule
│   ├── model.yaml                 # Model architecture
│   └── training.yaml              # Training hyperparameters
│
├── scripts/                       # Utility scripts
│   ├── download_dataset.py        # ✅ Download ARC data
│   ├── analyze_tasks.py           # ✅ Analyze task skills
│   ├── generate_curriculum.py     # Generate curriculum tasks
│   ├── train.py                   # Training script
│   └── evaluate.py                # Evaluation script
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_skill_development.ipynb
│   └── 03_curriculum_analysis.ipynb
│
├── CURRICULUM.md                  # ✅ Curriculum design
├── README.md                      # ✅ Project overview
└── requirements.txt               # Python dependencies
```

---

## Core Components

### 1. Skill Modules

Each skill is implemented as a separate module with a standard interface:

```python
class SkillModule:
    def forward(self, grid: Grid, context: Context) -> SkillOutput:
        """Apply skill to grid."""
        pass
    
    def train(self, task: Task) -> Loss:
        """Train on curriculum task."""
        pass
    
    def evaluate(self, task: Task) -> Metrics:
        """Evaluate skill performance."""
        pass
```

### 2. Task Generator

Generates synthetic tasks for each curriculum skill:

```python
class TaskGenerator:
    def generate_object_cognition_tasks(n: int) -> List[Task]:
        """Generate tasks requiring object cognition."""
        pass
    
    def generate_geometry_tasks(n: int) -> List[Task]:
        """Generate tasks requiring geometric reasoning."""
        pass
    
    # ... one generator per skill ...
```

### 3. Curriculum Scheduler

Manages progression through curriculum:

```python
class CurriculumScheduler:
    def get_current_stage(self) -> Stage:
        """Return current curriculum stage."""
        pass
    
    def get_next_batch(self) -> Batch:
        """Sample tasks for current stage."""
        pass
    
    def should_advance(self, metrics: Metrics) -> bool:
        """Check if ready to advance to next stage."""
        pass
```

### 4. Skill Composer

Combines skills to solve complex tasks:

```python
class SkillComposer:
    def decompose_task(self, task: Task) -> List[Skill]:
        """Identify which skills are needed."""
        pass
    
    def compose_solution(self, skills: List[Skill], task: Task) -> Solution:
        """Combine skills to solve task."""
        pass
```

### 5. ARC Solver

Main solver that uses skills to solve ARC tasks:

```python
class ARCSolver:
    def solve(self, task: ARCTask) -> Solution:
        """
        1. Analyze task to identify required skills
        2. Compose skills into a solution strategy
        3. Execute strategy to produce output
        4. Verify solution against test cases
        """
        pass
```

---

## Model Architecture

### Grid Encoder
- Input: 2D grid (up to 30x30, 10 colors)
- Output: Latent representation capturing spatial structure

### Reasoning Module  
- Multiple skill-specific sub-modules
- Attention mechanism to select relevant skills
- Compositional reasoning (combine skills)

### Grid Decoder
- Input: Latent representation + goal specification
- Output: Predicted output grid

---

## Training Strategy

### Phase 1: Individual Skill Training
- Train each skill module independently
- Use skill-specific curriculum tasks
- Achieve >90% accuracy on each skill before advancing

### Phase 2: Skill Composition
- Train on tasks requiring 2-3 skills
- Learn to chain/combine operations
- Develop skill selection strategy

### Phase 3: Complex Reasoning
- Train on tasks requiring 3+ skills
- Learn meta-reasoning (when to use which skill)
- Develop search and planning

### Phase 4: ARC Transfer
- Evaluate on ARC training set
- Fine-tune skill composition
- Optimize for ARC-specific patterns

### Phase 5: Final Evaluation
- Test on held-out ARC evaluation set
- Measure generalization performance
- Analyze failure modes

---

## Evaluation Metrics

### Skill-Level Metrics
- **Accuracy**: % of curriculum tasks solved correctly
- **Generalization**: Performance on novel variations
- **Efficiency**: Computational cost per task
- **Robustness**: Performance under noise/perturbations

### Task-Level Metrics
- **Solve Rate**: % of ARC tasks solved
- **Attempt Efficiency**: Average attempts needed
- **Skill Coverage**: Which skills are actually used
- **Error Analysis**: Why failures occur

### Meta-Level Metrics
- **Data Efficiency**: Performance vs. training examples
- **Transfer**: Curriculum → ARC performance gap
- **Interpretability**: Clarity of reasoning traces
- **Novelty Handling**: Performance on unusual tasks

---

## Technology Stack

### Core
- **Python 3.10+**: Main language
- **PyTorch 2.0+**: Deep learning framework
- **NumPy**: Grid operations

### Training
- **PyTorch Lightning**: Training infrastructure
- **WandB**: Experiment tracking
- **Hydra**: Configuration management

### Utilities
- **Matplotlib/Seaborn**: Visualization
- **Pandas**: Data analysis
- **Pytest**: Testing

---

## Development Roadmap

### Week 1-2: Foundation ✅
- [x] Define curriculum (CURRICULUM.md)
- [x] Download ARC dataset
- [x] Analyze task skill requirements
- [ ] Set up project structure
- [ ] Implement grid utilities

### Week 3-4: Core Skills
- [ ] Implement object cognition module
- [ ] Implement numerosity module
- [ ] Implement geometry module
- [ ] Generate curriculum tasks for core skills
- [ ] Train and evaluate core skills

### Week 5-6: Cognitive Operations
- [ ] Implement pattern recognition
- [ ] Implement transformations
- [ ] Implement analogy reasoning
- [ ] Generate curriculum tasks
- [ ] Train and evaluate

### Week 7-8: Skill Composition
- [ ] Implement skill composer
- [ ] Generate multi-skill tasks
- [ ] Train composition system
- [ ] Evaluate on complex curriculum tasks

### Week 9-10: ARC Integration
- [ ] Integrate with ARC tasks
- [ ] Implement full solver pipeline
- [ ] Tune and optimize
- [ ] Final evaluation on ARC eval set

---

## Success Criteria

**Minimum Viable Performance**:
- Each skill module: >85% on curriculum tasks
- Skill composition: >70% on 2-skill tasks
- ARC training set: >30% solve rate

**Target Performance**:
- Each skill module: >95% on curriculum tasks
- Skill composition: >85% on 3-skill tasks  
- ARC training set: >50% solve rate
- ARC eval set: >40% solve rate

**Stretch Goal**:
- ARC eval set: >60% solve rate (human-level)

---

## Next Steps

1. **Verify dataset download** - Ensure all 800 tasks downloaded
2. **Run task analysis** - Validate curriculum against real ARC tasks
3. **Build project structure** - Create directories and base classes
4. **Implement grid utilities** - Core operations for grid manipulation
5. **Start with object cognition** - First curriculum module

The foundation is ready. Now we build! 🚀
