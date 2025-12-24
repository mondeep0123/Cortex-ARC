# ARC-AGI Primitives - Organized Structure

## Overview

This directory contains all 5 core primitives for ARC reasoning, each in their own organized subdirectory.

```
primitives/
├── object_cognition/     ✅ COMPLETE (100% IoU)
├── numerosity/           🚧 IN PROGRESS
├── geometry/             ⏳ TODO
├── topology/             ⏳ TODO
└── physics/              ⏳ TODO
```

## Primitives

### 1. Object Cognition ✅
**Status**: Complete  
**Purpose**: Segment objects from background  
**Performance**: 100% IoU on all benchmarks  
**Files**: 4 core files + checkpoint

[View Details](./object_cognition/README.md)

### 2. Numerosity 🚧
**Status**: In Development  
**Purpose**: Counting & comparison  
**Expected**: 95%+ accuracy  
**Files**: Primitive + Curriculum ready to train

[View Spec](./numerosity/SPEC.md)

### 3. Geometry ⏳
**Status**: Not Started  
**Purpose**: Shapes, symmetry, rotation  
**Planned**: U-Net + geometric features

### 4. Topology ⏳
**Status**: Not Started  
**Purpose**: Containment, connectivity  
**Planned**: Graph neural features

### 5. Physics ⏳
**Status**: Not Started  
**Purpose**: Gravity, support, layers  
**Planned**: Spatial reasoning network

## Training Pipeline

Each primitive follows this structure:

```
primitives/PRIMITIVE_NAME/
├── PRIMITIVE_NAME_primitive.py    # Model implementation
├── curriculum_PRIMITIVE_NAME.py   # Training data generator
├── benchmark_PRIMITIVE_NAME.py    # Handcrafted tests
├── README.md or SPEC.md           # Documentation
└── (results/)                     # Results and analysis
```

## Shared Infrastructure

Located in `src/primitives/`:
- `base_primitive.py` - Base class for all primitives
- `__init__.py` - Module exports

## Training Scripts

Root level:
- `train_hp_primitive.py` - High-performance training script
- `configs/high_performance.yaml` - Training configuration

## Key Learnings (from Object Cognition)

### ✅ Do This
1. **Spatial Preservation**: Use conv layers, preserve dimensions
2. **U-Net Architecture**: Skip connections work perfectly
3. **Clear Task Definition**: One primitive = one clear task
4. **Consistent Rules**: Same rules in curriculum and test
5. **MSE for Regression**: Prevents "predict max" shortcuts

### ❌ Avoid This  
1. Flattening spatial data
2. Classification for continuous values (counting)
3. Ambiguous task definitions
4. Curriculum-test mismatches
5. Weak supervision signals

## Progress Tracker

| Primitive | Architecture | Curriculum | Training | Eval | Status |
|-----------|-------------|------------|----------|------|--------|
| Object Cognition | ✅ U-Net | ✅ Done | ✅ 100% | ✅ 100% | **COMPLETE** |
| Numerosity | ✅ Spatial | ✅ Done | ⏳ Ready | ⏳ Pending | **READY TO TRAIN** |
| Geometry | ⏳ Design | ⏳ Plan | ⏳ | ⏳ | TODO |
| Topology | ⏳ Design | ⏳ Plan | ⏳ | ⏳ | TODO |
| Physics | ⏳ Design | ⏳ Plan | ⏳ | ⏳ | TODO |

## Timeline

- **Object Cognition**: ✅ Complete (Dec 24, 2024)
- **Numerosity**: 🎯 Target: Today (Dec 24, 2024)
- **Geometry**: Target: Dec 25, 2024
- **Topology**: Target: Dec 25, 2024
- **Physics**: Target: Dec 26, 2024
- **Integration**: Target: Dec 27, 2024

## Next Steps

1. ✅ Organize Object Cognition files
2. ✅ Define Numerosity clearly
3. ✅ Create Numerosity architecture
4. ✅ Create Numerosity curriculum
5. ⏳ **Train Numerosity** ← YOU ARE HERE
6. ⏳ Evaluate Numerosity
7. ⏳ Move to Geometry

---

*Updated: December 24, 2024, 3:55 AM IST*
