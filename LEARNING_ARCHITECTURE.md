# Learning Architecture Decision

## The Critical Question
**"Has the model actually LEARNED object cognition, or is it just hard-coded rules?"**

This is the RIGHT question to ask. Let me be completely transparent.

---

## Current State: Rule-Based (No Learning)

### What `object_cognition.py` Actually Is
```python
# Current implementation
def find_objects(grid):
    # Uses flood-fill algorithm ← HARD-CODED
    # No parameters to learn
    # No training required
    # No overfitting possible (because no fitting!)
```

**Reality Check**:
- ❌ No neural network
- ❌ No trainable parameters
- ❌ No gradient descent
- ❌ No learning at all
- ✅ Deterministic algorithm
- ✅ 100% reproducible

### Why This Isn't "Learning"
The current system is like a calculator:
- Calculator doesn't "learn" arithmetic
- It executes hard-coded algorithms
- Our object detection is the same

---

## Three Architecture Options

### Option 1: Pure Algorithmic (Current)

**Implementation**:
```python
class AlgorithmicObjectCognition:
    def detect_objects(self, grid):
        # Flood-fill algorithm
        # Manually coded logic
        return connected_components
```

**Pros**:
- ✅ Reliable and deterministic
- ✅ No training data needed
- ✅ No overfitting possible
- ✅ Fully interpretable
- ✅ Fast and efficient

**Cons**:
- ❌ Cannot adapt to novel object definitions
- ❌ Brittle to edge cases
- ❌ Limited to pre-programmed rules
- ❌ No generalization beyond rules

**When to Use**:
- Well-defined primitives (connected components, symmetry, etc.)
- Reliability > flexibility
- Limited training data

---

### Option 2: Pure Neural Learning (New Implementation)

**Implementation**:
```python
class NeuralObjectCognition(nn.Module):
    def __init__(self):
        self.encoder = CNN(...)  # Learns object features
        self.decoder = CNN(...)  # Learns to manipulate
    
    def forward(self, grid):
        # Everything is learned from data
        features = self.encoder(grid)
        output = self.decoder(features)
        return output
```

**Training Process**:
1. Generate curriculum tasks (diverse object detection scenarios)
2. Train encoder-decoder on tasks
3. Validate on held-out tasks to prevent overfitting
4. Evaluate generalization to novel object types

**Preventing Overfitting**:
- **Curriculum diversity**: Many task variations, not just ARC
- **Train/Val/Test split**: Never test on training data
- **Regularization**: Dropout, weight decay, batch norm
- **Early stopping**: Stop when validation loss increases
- **Data augmentation**: Rotate, flip, color-swap tasks

**Pros**:
- ✅ Can adapt to novel patterns
- ✅ Learns from curriculum tasks
- ✅ Discovers object concepts
- ✅ True generalization possible

**Cons**:
- ❌ Needs lots of training data
- ❌ Computationally expensive
- ❌ Can overfit if not careful
- ❌ Less interpretable
- ❌ Non-deterministic

**When to Use**:
- Plenty of training data available
- Flexibility > reliability
- Concept learning needed

---

### Option 3: Neuro-Symbolic Hybrid (RECOMMENDED) ⭐

**Implementation**:
```python
class HybridObjectCognition:
    def __init__(self):
        # Algorithmic primitives
        self.flood_fill =  AlgorithmicFloodFill()
        self.symmetry_detector = AlgorithmicSymmetry()
        
        # Learned components
        self.skill_selector = NeuralSkillSelector()  # Learns WHEN
        self.parameter_predictor = NeuralParams()     # Learns HOW
        self.composer = NeuralComposer()              # Learns WHICH
    
    def apply(self, grid, task_context):
        # 1. Neural network decides which primitive to use
        selected_skill = self.skill_selector(grid, task_context)
        
        # 2. Neural network predicts parameters
        params = self.parameter_predictor(grid)
        
        # 3. Execute reliable algorithm with learned params
        if selected_skill == "detect_objects":
            result = self.flood_fill(grid, connectivity=params.connectivity)
        
        # 4. Neural composer decides how to combine results
        final = self.composer([result], task_context)
        
        return final
```

**What's Learned**:
- ✅ Which skill to apply (skill selection)
- ✅ How to parameterize skills (parameter prediction)
- ✅ How to compose skills (composition strategy)

**What's Algorithmic**:
- ✅ Connected components (flood-fill)
- ✅ Symmetry detection
- ✅ Geometric transformations
- ✅ Color mapping

**Pros**:
- ✅ Best of both worlds
- ✅ Reliable primitives + flexible composition
- ✅ Data-efficient (learn selection, not primitives)
- ✅ Interpretable (clear reasoning traces)
- ✅ Less prone to overfitting (fewer parameters)

**Cons**:
- ⚠️ More complex architecture
- ⚠️ Requires careful design
- ⚠️ Need curriculum for selection learning

**When to Use**:
- Limited training data (like ARC!)
- Need both reliability AND flexibility
- Interpretability is important
- Production systems

---

## Preventing Overfitting in Each Approach

### Algorithmic (Current)
**Overfitting**: Not applicable
- No parameters = no overfitting
- Can be too rigid instead

### Pure Neural
**Overfitting Prevention**:
```python
# 1. Diverse curriculum tasks (NOT just ARC)
curriculum_tasks = generate_curriculum(
    num_tasks=10000,
    variations=[
        "object_detection",
        "object_counting", 
        "object_extraction",
        "color_filtering",
        # ... many more
    ]
)

# 2. Train/Val/Test split
train_tasks = curriculum_tasks[:7000]
val_tasks = curriculum_tasks[7000:8500]  # For early stopping
test_tasks = curriculum_tasks[8500:]      # For final evaluation

# 3. Training with validation
for epoch in range(max_epochs):
    train_loss = train_on_batch(train_tasks)
    val_loss = evaluate(val_tasks)
    
    if val_loss > best_val_loss:
        epochs_without_improvement += 1
        if epochs_without_improvement > patience:
            break  # Early stopping
    
    # 4. Regularization
    loss = task_loss + 0.01 * weight_decay + dropout_loss

# 5. Final check for overfitting
test_metrics = evaluate(test_tasks)
if train_acc - test_acc > 0.2:  # >20% gap
    print("WARNING: Model is overfitting!")
```

### Hybrid (Recommended)
**Overfitting Prevention**:
- Fewer parameters to learn (only selection/composition)
- Algorithmic primitives provide inductive bias
- Curriculum focuses on skill composition, not low-level features
- Easier to regularize (smaller networks)

---

## What We Should Build: Hybrid Architecture

### Phase 1: Algorithmic Primitives (Current) ✅
```
Skills: [object_detection, symmetry, counting, ...]
Implementation: Algorithms
Status: DONE for object_detection
```

### Phase 2: Neural Skill Selector (Next)
```python
class SkillSelector(nn.Module):
    """Learns WHICH skill to apply to a task."""
    
    def forward(self, task_examples):
        # Encode task demonstration pairs
        task_encoding = self.encode_task(task_examples)
        
        # Predict skill relevance scores
        skill_scores = self.score_skills(task_encoding)
        # → [0.9, 0.1, 0.3, ...]  for each skill
        
        return skill_scores
```

**Training**:
- Curriculum: Tasks labeled with required skills
- Loss: Cross-entropy on skill labels
- Prevents overfitting: Many diverse task types

### Phase 3: Neural Composer
```python
class SkillComposer(nn.Module):
    """Learns HOW to combine skills."""
    
    def forward(self, task_examples, available_skills):
        # Learn skill sequence: [skill_A, skill_B, skill_C]
        composition_plan = self.plan_composition(
            task_examples,
            available_skills
        )
        return composition_plan
```

**Training**:
- Curriculum: Multi-skill tasks
- Loss: Sequence prediction + final output correctness
- Prevents overfitting: Compositional structure, not memorization

---

## Generalization Test Design

### Test 1: Within-Distribution
Train on 80% of curriculum tasks, test on 20%
- **Pass**: >90% accuracy on held-out tasks
- **Fail**: <70% accuracy → overfitting

### Test 2: Novel Variations
Train on "detect red objects", test on "detect blue objects"
- **Pass**: >80% transfer
- **Fail**: <50% transfer → memorization,not learning

### Test 3: Composition
Train on single-skill tasks, test on 2-skill combinations
- **Pass**: >70% composition
- **Fail**: <40% → no compositional generalization

### Test 4: ARC Transfer (Final)
Train on curriculum (NO ARC), test on ARC
- **Pass**: >40% on ARC eval
- **Fail**: <20% → curriculum doesn't transfer

---

## Implementation Roadmap

### Week 1-2: Algorithmic Primitives
- [x] Object detection (flood-fill)
- [ ] Symmetry detection
- [ ] Counting utilities
- [ ] Geometric transforms
- [ ] Topology operations

### Week 3-4: Curriculum Generation
- [ ] Object detection tasks (10K variations)
- [ ] Symmetry tasks
- [ ] Counting tasks
- [ ] Multi-skill tasks
- Train/val/test split for each

### Week 5-6: Neural Selector
- [ ] Task encoder network
- [ ] Skill scoring network
- [ ] Train on labeled curriculum
- [ ] Validate generalization

### Week 7-8: Neural Composer
- [ ] Composition planning network
- [ ] Sequence prediction
- [ ] Train on multi-skill tasks
- [ ] Test compositional generalization

### Week 9-10: Integration & ARC Eval
- [ ] Full hybrid system
- [ ] ARC transfer evaluation
- [ ] Error analysis
- [ ] Iterative improvement

---

## Answer to Your Question

> "Is it really learned object cognition, perfectly, sure no overfitting?"

**Current State (object_cognition.py)**:
- ❌ No, not learned - it's algorithmic
- ✅ No overfitting (but also no fitting)
- ⚠️ Reliable but inflexible

**Recommended Path (Hybrid)**:
- ✅ Learn skill selection & composition
- ✅ Use algorithms for primitives
- ✅ Prevent overfitting through:
  - Diverse curriculum (not ARC-specific)
  - Validation sets
  - Regularization
  - Compositional architecture
  - Transfer testing

**Timeline to "True Learning"**:
- Current: Algorithms only (Week 1)
- Phase 2: Learn selection (Week 6)
- Phase 3: Learn composition (Week 8)
- Full system: Hybrid with learned reasoning (Week 10)

---

## The Path Forward

I recommend we:

1. **Keep algorithmic primitives** (current object_cognition.py)
   - Reliable foundation
   - No training needed
   - Good baseline

2. **Add neural skill selector** (next milestone)
   - Learns WHICH skill for WHICH task
   - Trained on diverse curriculum
   - Validation prevents overfitting

3. **Add neural composer** (after selector works)
   - Learns HOW to combine skills
   - Compositional generalization
   - Transfer to ARC

This hybrid approach gives us:
- **Reliability** from algorithms
- **Flexibility** from learning
- **Interpretability** from explicit reasoning
- **Generalization** from compositional structure

Does this address your concerns about learning and overfitting?
