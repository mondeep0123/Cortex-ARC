# Numerosity Primitive - Specification

**Status**: 🚧 **IN DEVELOPMENT**  
**Start Date**: December 24, 2024

## Purpose

**Numerosity**: Understanding quantities, counting, and numerical relationships.

**Core Capabilities**:
1. **Counting**: How many objects/colors/patterns?
2. **Comparison**: More than? Less than? Equal?
3. **Arithmetic**: Simple add/subtract (if needed)

## Task Definition

Given a grid, answer numerical questions:
- "How many objects are there?" → Count
- "How many cells are color X?" → Count by property
- "Is group A larger than group B?" → Compare
- "What's the most common color?" → Mode/Max

## Architecture Decision

**Lesson from Object Cognition**: Use spatial-preserving architecture!

### Proposed: Spatial Counting Network

```
Input Grid [H, W]
    ↓
Color Embedding [H, W, 32]
    ↓
Spatial Encoder (conv layers)
  - Preserve spatial structure
  - Extract counting features
    ↓
Global Aggregation
  - Count pooling per color/object
  - Spatial attention
    ↓
Count Heads
  - Total count
  - Per-color counts
  - Comparison outputs
    ↓
Outputs:
  - total_count: [batch, 1] (0-20 range)
  - color_counts: [batch, 10] (count per color)
  - comparison: [batch, 3] (less/equal/more)
```

**Key Insight**: Use Object Cognition output as input!
- Object Cognition gives us WHERE objects are
- Numerosity counts HOW MANY

## Training Strategy

### Curriculum Tasks

1. **Simple Counting** (Easy)
   - Empty grid → 0
   - Single object → 1
   - Multiple separate objects → N

2. **Color Counting** (Medium)
   - Count red objects
   - Count blue cells
   - Find most common color

3. **Comparison** (Medium)
   - More red or blue?
   - Equal counts?
   - Largest group?

4. **Complex** (Hard)
   - Count connected components
   - Count by size (large vs small)
   - Count by pattern

### Target Metrics

- **Training**: 95%+ accuracy
- **Validation**: 95%+ accuracy  
- **Test**: 95%+ accuracy
- **Handcrafted**: 90%+ accuracy

**Goal**: Match Object Cognition's success (but counting is harder than segmentation)

## Implementation Plan

### Phase 1: Architecture ✅
- [x] Define task clearly
- [ ] Design spatial counting network
- [ ] Implement model
- [ ] Test on single example

### Phase 2: Curriculum ⏳
- [ ] Generate simple counting tasks
- [ ] Generate color counting tasks
- [ ] Generate comparison tasks
- [ ] Create balanced dataset

### Phase 3: Training ⏳
- [ ] Train with U-Net lessons applied
- [ ] Monitor convergence
- [ ] Validate on curriculum
- [ ] Test on handcrafted

### Phase 4: Evaluation ⏳
- [ ] Create handcrafted benchmark
- [ ] Evaluate transfer
- [ ] Compare to baseline
- [ ] Document results

## Key Differences from Object Cognition

| Aspect | Object Cognition | Numerosity |
|--------|-----------------|------------|
| **Output** | Spatial mask [H,W] | Scalar counts |
| **Task** | Segmentation (per-pixel) | Regression/Classification |
| **Aggregation** | None | Global pooling |
| **Input** | Raw grid | Grid + object masks |
| **Loss** | Binary CE (segmentation) | MSE/CE (counting) |

## Avoiding Past Mistakes

❌ **Don't**: Use classification for counting (0-15 classes)
  - Leads to "predict max" shortcut
  
✅ **Do**: Use regression with proper loss
  - Predict continuous value
  - Round to integer
  - MSE loss prevents shortcuts

❌ **Don't**: Flatten and lose spatial info
  - Can't see WHERE to count
  
✅ **Do**: Keep spatial structure
  - Use conv layers
  - Aggregate globally at the end

❌ **Don't**: Mismatch curriculum and test
  - Different counting rules
  
✅ **Do**: Consistent definition
  - Clear rule: "count non-background objects"
  - Same rule everywhere

## Dependencies

- **Object Cognition** (completed) - provides object masks
- Can work standalone too (count from raw grid)

## Success Criteria

- [ ] 95%+ on simple counting
- [ ] 90%+ on color counting
- [ ] 90%+ on comparisons
- [ ] Transfers to handcrafted tests
- [ ] No shortcuts (verified through analysis)

## Timeline

- **Architecture**: 1 hour
- **Curriculum**: 1 hour  
- **Training**: 2-3 hours
- **Evaluation**: 1 hour
- **Total**: ~6 hours

**Expected completion**: December 24, 2024 (same day!)

---

*Let's get this right the first time! 🚀*
