# Primitive Learning System - Setup Complete

## Overview
System for training specialist primitive models, testing generalization, then distilling into student.

## Architecture

### 5 Core Primitives (Small Specialist Models)
Each ~50K-100K parameters

1. **Object Cognition** ✅ IMPLEMENTED
   - Detects, counts, segments objects
   - File: `src/primitives/object_cognition_primitive.py`
   - Curriculum: `src/primitives/curriculum_object_cognition.py`
   - Training: `train_primitive_1_object_cognition.py`

2. **Numerosity** 📋 TODO
   - Counting, comparison, arithmetic
   - Tasks: count objects, compare quantities, add/subtract
   - File: `src/primitives/numerosity_primitive.py`

3. **Geometry** 📋 TODO
   - Shapes, symmetry, rotation, scaling
   - Tasks: detect symmetry, identify shapes, apply rotations
   - File: `src/primitives/geometry_primitive.py`

4. **Topology** 📋 TODO
   - Containment, connectivity, proximity
   - Tasks: inside/outside, connected components, adjacency
   - File: `src/primitives/topology_primitive.py`

5. **Physics** 📋 TODO
   - Gravity, support, occlusion, layers
   - Tasks: simulate falling, stack stability, layer ordering
   - File: `src/primitives/physics_primitive.py`

### Student Model (Knowledge Distillation)
After all primitives generalize well:
- Single model that learns from all 5 teacher primitives
- Smaller than combined primitives (~200K params total)
- File: `src/student/unified_primitive_model.py`

## Training Pipeline

### For Each Primitive:

```
1. Generate Curriculum (10K tasks)
   ├── Train set: 7,000 tasks
   ├── Val set: 1,500 tasks (early stopping)
   └── Test set: 1,500 tasks (NEVER seen during training)

2. Train with Early Stopping
   ├── Monitor validation loss
   ├── Save best model
   └── Stop if no improvement for 10 epochs

3. Test Generalization
   ├── Evaluate on test set
   ├── Measure train-test gap
   └── Decision:
       ├── Gap < 20% AND test_acc > 70% → ✅ GENERALIZES
       └── Otherwise → ❌ NEEDS IMPROVEMENT

4. If Generalizes:
   ├── Save primitive model
   ├── Mark as ready for distillation
   └── Proceed to next primitive

5. If Doesn't Generalize:
   ├── Analyze failure mode
   ├── Improve curriculum OR model OR regularization
   └── Retrain
```

### Knowledge Distillation (After all 5 ready):

```
1. Load all 5 trained primitives
2. Generate mixed curriculum (all skills)
3. Student model learns to mimic all 5 teachers
4. Test student on:
   ├── Each primitive's test set
   ├── Mixed multi-skill tasks
   └── Finally: ARC benchmark
```

## Files Created

### Core Framework
- ✅ `src/primitives/base_primitive.py` - Base class for all primitives
- ✅ `src/primitives/object_cognition_primitive.py` - Primitive #1
- ✅ `src/primitives/curriculum_object_cognition.py` - Curriculum generator #1
- ✅ `train_primitive_1_object_cognition.py` - Training script #1
- ✅ `PRIMITIVES_SETUP.md` - This file

### To Be Created (Templates)
- 📋 `src/primitives/numerosity_primitive.py`
- 📋 `src/primitives/curriculum_numerosity.py`
- 📋 `train_primitive_2_numerosity.py`
- 📋 (Same for Geometry, Topology, Physics)
- 📋 `src/student/unified_primitive_model.py`
- 📋 `distill_into_student.py`

## How to Train Primitive #1

```bash
# Train object cognition primitive
python train_primitive_1_object_cognition.py

# This will:
# 1. Generate 10K curriculum tasks
# 2. Train with early stopping
# 3. Test generalization
# 4. Save model if it generalizes well
```

## Generalization Criteria

A primitive is considered to "generalize well" if:
1. **Test Accuracy > 70%** - Can solve unseen tasks
2. **Train-Test Gap < 20%** - Not overfitting
3. **Val-Test Gap < 10%** - Validation is representative

If these are met → ✅ Ready for knowledge distillation

## Next Steps

1. **Run training for Primitive #1**:
   ```bash
   python train_primitive_1_object_cognition.py
   ```

2. **If it generalizes** (test acc > 70%, gap < 20%):
   - ✅ Mark Object Cognition as complete
   - Move to Primitive #2 (Numerosity)

3. **If it doesn't generalize**:
   - Analyze why (check results/object_cognition_generalization.json)
   - Improve curriculum (more diversity)
   - OR increase model capacity
   - OR add more regularization
   - Retrain

4. **Repeat for all 5 primitives**

5. **Once all 5 generalize**:
   - Create student model
   - Distill all 5 into student
   - Test student on ARC

## Expected Timeline

- **Primitive #1 (Object Cognition)**: Training ~10-30 min (CPU) or ~2-5 min (GPU)
- **Primitive #2-5**: ~10-30 min each
- **Student Distillation**: ~1-2 hours
- **Total**: ~2-4 hours for complete system

## Success Metrics

### Per Primitive:
- Test Accuracy: >70%
- Generalization Gap: <20%
- Model Size: <100K parameters

### Student Model:
- Average across primitives: >65%
- Multi-skill composition: >60%
- **ARC Transfer**: >30% (ultimate test!)

## Why This Approach Works

1. **Modularity**: Each primitive is independent
2. **Testability**: Can verify generalization per skill
3. **Interpretability**: Know which skills the model has
4. **Compositionality**: Student learns to combine primitives
5. **Data Efficiency**: Small models need less data

## Architecture Diagram

```
Primitive 1           Primitive 2          Primitive 3          Primitive 4          Primitive 5
[Object Cognition] → [Numerosity]      → [Geometry]        → [Topology]        → [Physics]
    64 hidden            64 hidden          64 hidden           64 hidden           64 hidden
    ~50K params          ~50K params        ~50K params         ~50K params         ~50K params
    
    Train on 7K          Train on 7K        Train on 7K         Train on 7K         Train on 7K
    Test Gen? ✓          Test Gen? ✓        Test Gen? ✓         Test Gen? ✓         Test Gen? ✓
         ↓                    ↓                   ↓                   ↓                   ↓
         └────────────────────┴───────────────────┴───────────────────┴───────────────────┘
                                           ↓
                                 [Student Model]
                                  128 hidden dim
                                  ~200K params
                                  
                            Learns from all 5 teachers
                            Tests on multi-skill tasks
                                   Finally → ARC
```

## Current Status

- ✅ Framework: Complete
- ✅ Primitive #1: Code ready, needs training
- ⏳ Primitive #2-5: Templates to create
- ⏳ Student Model: After all primitives ready
- ⏳ ARC Evaluation: Final step

## Ready to Begin!

Run this command to start:
```bash
python train_primitive_1_object_cognition.py
```

Watch for the generalization report at the end!
