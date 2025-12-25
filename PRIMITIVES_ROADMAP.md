# 🧠 Primitives Roadmap: ARC-AGI 1 & 2

> A comprehensive list of cognitive primitives needed to solve ARC-AGI benchmarks.
> Based on Chollet's Core Knowledge priors and extended for ARC-AGI 2.

**Last Updated:** December 26, 2025

---

## 📊 Progress Overview

| Category | Total | Completed | In Progress | Planned |
|----------|-------|-----------|-------------|---------|
| Core Primitives (1-13) | 13 | 2 | 0 | 11 |
| Extended Skills (14-30) | 17 | 0 | 0 | 17 |
| **Total** | **30** | **2** | **0** | **28** |

**Current Accuracy:** 2/30 primitives at 100%

---

## ✅ COMPLETED PRIMITIVES

### 1. Object Cognition ✅
| Attribute | Value |
|-----------|-------|
| **Status** | ✅ COMPLETE |
| **Accuracy** | 100% IoU |
| **Date** | December 24, 2025 |
| **File** | `train_primitive_1_object_cognition.py` |
| **Method** | U-Net with skip connections |
| **Description** | Segments foreground objects from background (color 0) |

**Capabilities:**
- Binary mask generation
- Foreground/background separation
- Spatial preservation via skip connections

---

### 2. Numerosity ✅
| Attribute | Value |
|-----------|-------|
| **Status** | ✅ COMPLETE |
| **Accuracy** | 100% |
| **Date** | December 25, 2025 (Christmas!) |
| **File** | `train_staged.py` |
| **Method** | Staged Training (Subitizing + Pure Arithmetic) |
| **Description** | Counts total non-zero pixels in a grid |

**Capabilities:**
- Count 0-30+ objects
- Subitizing (instant recognition of 0-4)
- Exact addition via memorized lookup

---

## 🔄 CORE PRIMITIVES (Chollet's Priors)

### 3. Color Understanding 🔴
| Attribute | Value |
|-----------|-------|
| **Status** | 🔴 PLANNED |
| **Priority** | HIGH |
| **Description** | Identify, compare, and manipulate colors |

**Sub-capabilities needed:**
- [ ] Identify dominant color
- [ ] Identify rarest color
- [ ] Count specific color
- [ ] Mask by color (keep only red, etc.)
- [ ] Color frequency histogram
- [ ] Color swapping rules

---

### 4. Geometry 🔴
| Attribute | Value |
|-----------|-------|
| **Status** | 🔴 PLANNED |
| **Priority** | HIGH |
| **Description** | Recognize and manipulate shapes, lines, regions |

**Sub-capabilities needed:**
- [ ] Detect lines (horizontal, vertical, diagonal)
- [ ] Detect rectangles
- [ ] Detect squares
- [ ] Detect L-shapes, T-shapes
- [ ] Calculate area
- [ ] Calculate perimeter
- [ ] Detect corners

---

### 5. Topology 🔴
| Attribute | Value |
|-----------|-------|
| **Status** | 🔴 PLANNED |
| **Priority** | HIGH |
| **Description** | Understand connectivity, containment, and boundaries |

**Sub-capabilities needed:**
- [ ] Connected component detection
- [ ] Inside/outside determination
- [ ] Hole detection
- [ ] Boundary/edge detection
- [ ] Enclosure detection
- [ ] Path connectivity

---

### 6. Symmetry 🔴
| Attribute | Value |
|-----------|-------|
| **Status** | 🔴 PLANNED |
| **Priority** | MEDIUM |
| **Description** | Detect and apply symmetry operations |

**Sub-capabilities needed:**
- [ ] Horizontal reflection symmetry
- [ ] Vertical reflection symmetry
- [ ] Rotational symmetry (90°, 180°, 270°)
- [ ] Point symmetry
- [ ] Symmetry completion (fill missing parts)

---

### 7. Spatial Relationships 🔴
| Attribute | Value |
|-----------|-------|
| **Status** | 🔴 PLANNED |
| **Priority** | HIGH |
| **Description** | Understand relative positions of objects |

**Sub-capabilities needed:**
- [ ] Above/below
- [ ] Left/right
- [ ] Inside/outside
- [ ] Adjacent/touching
- [ ] Distance measurement
- [ ] Alignment detection

---

### 8. Pattern Recognition 🔴
| Attribute | Value |
|-----------|-------|
| **Status** | 🔴 PLANNED |
| **Priority** | HIGH |
| **Description** | Identify repeating patterns and regularities |

**Sub-capabilities needed:**
- [ ] Repeating horizontal patterns
- [ ] Repeating vertical patterns
- [ ] Tiling patterns
- [ ] Sequence detection
- [ ] Pattern period detection
- [ ] Pattern extrapolation

---

### 9. Transformation 🔴
| Attribute | Value |
|-----------|-------|
| **Status** | 🔴 PLANNED |
| **Priority** | MEDIUM |
| **Description** | Apply geometric transformations |

**Sub-capabilities needed:**
- [ ] Rotation (90°, 180°, 270°)
- [ ] Reflection (horizontal, vertical)
- [ ] Scaling (2x, 3x, shrink)
- [ ] Translation (shift)
- [ ] Transpose

---

### 10. Part-Whole Relationships 🔴
| Attribute | Value |
|-----------|-------|
| **Status** | 🔴 PLANNED |
| **Priority** | MEDIUM |
| **Description** | Understand objects composed of parts |

**Sub-capabilities needed:**
- [ ] Object decomposition
- [ ] Part identification
- [ ] Hierarchical structure
- [ ] Assembly/disassembly

---

### 11. Intuitive Physics 🔴
| Attribute | Value |
|-----------|-------|
| **Status** | 🔴 PLANNED |
| **Priority** | LOW |
| **Description** | Basic physical intuitions |

**Sub-capabilities needed:**
- [ ] Gravity (objects fall down)
- [ ] Support (objects rest on surfaces)
- [ ] Contact
- [ ] Occlusion (objects behind others)

---

### 12. Agents/Goals 🔴
| Attribute | Value |
|-----------|-------|
| **Status** | 🔴 PLANNED |
| **Priority** | LOW |
| **Description** | Directed, goal-seeking behavior |

**Sub-capabilities needed:**
- [ ] Path to goal
- [ ] Obstacle avoidance
- [ ] Target reaching

---

### 13. Temporal/Sequential 🔴
| Attribute | Value |
|-----------|-------|
| **Status** | 🔴 PLANNED |
| **Priority** | MEDIUM |
| **Description** | Order, sequence, before/after |

**Sub-capabilities needed:**
- [ ] Sequence ordering
- [ ] Before/after
- [ ] Cause/effect
- [ ] Step-by-step operations

---

## 🚀 EXTENDED SKILLS FOR ARC-AGI 2

### 14. Analogy Making 🔴
| Priority | HARD |
|----------|------|
| **Description** | A:B :: C:? reasoning |
| **Why needed** | Many ARC puzzles require finding analogies |

---

### 15. Rule Induction 🔴
| Priority | HARD |
|----------|------|
| **Description** | Infer transformation rules from examples |
| **Why needed** | Core of ARC - figure out the rule from examples |

---

### 16. Compositional Generalization 🔴
| Priority | VERY HARD |
|----------|-----------|
| **Description** | Combine primitives in novel ways |
| **Why needed** | ARC-AGI 2 will require novel combinations |

---

### 17. Abstract Categorization 🔴
| Priority | HARD |
|----------|------|
| **Description** | Group by abstract properties, not surface features |
| **Why needed** | Generalization beyond specific examples |

---

### 18. Procedural Reasoning 🔴
| Priority | HARD |
|----------|------|
| **Description** | Multi-step algorithmic operations |
| **Why needed** | Complex transformations require sequences of steps |

---

### 19. Meta-Learning 🔴
| Priority | VERY HARD |
|----------|-----------|
| **Description** | Learn to learn from few examples |
| **Why needed** | ARC's few-shot nature requires rapid adaptation |

---

### 20. Program Synthesis 🔴
| Priority | VERY HARD |
|----------|-----------|
| **Description** | Generate code/programs from I/O examples |
| **Why needed** | Ultimate goal - express transformations as programs |

---

### 21-30. Additional Grid Skills 🔴

| # | Skill | Priority | Description |
|---|-------|----------|-------------|
| 21 | Spatial Transformation | MEDIUM | Complex rotations, reflections, warps |
| 22 | Grid Manipulation | MEDIUM | Resize, crop, pad, tile |
| 23 | Color Mapping | MEDIUM | Systematic recoloring rules |
| 24 | Object Tracking | HARD | Identify same object across transformations |
| 25 | Boundary Detection | MEDIUM | Edges, contours, perimeters |
| 26 | Fill/Flood | MEDIUM | Region filling, propagation |
| 27 | Masking/Selection | MEDIUM | Select by property |
| 28 | Grid Arithmetic | MEDIUM | Size comparisons, area calculations |
| 29 | Copying/Mirroring | MEDIUM | Replicate with modifications |
| 30 | Completion | HARD | Fill in missing pattern parts |

---

## 🎯 Recommended Implementation Order

### Phase 1: Foundation (Current)
1. ✅ Object Cognition
2. ✅ Numerosity

### Phase 2: Color & Spatial (Next)
3. 🔴 Color Understanding
4. 🔴 Spatial Relationships
5. 🔴 Geometry

### Phase 3: Structure
6. 🔴 Topology
7. 🔴 Pattern Recognition
8. 🔴 Symmetry

### Phase 4: Transformation
9. 🔴 Transformation
10. 🔴 Part-Whole
11. 🔴 Grid Manipulation

### Phase 5: Reasoning (Hard)
12. 🔴 Rule Induction
13. 🔴 Analogy Making
14. 🔴 Compositional Generalization

### Phase 6: Integration
15-30. Combined primitives, meta-controller, program synthesis

---

## 🧩 The Meta-Controller Problem

> **Critical Challenge:** ARC rules prohibit hardcoding which primitives to use.

The model must **learn** to:
1. Analyze a puzzle
2. Decide which primitives are relevant
3. Compose them in the correct order
4. Execute and produce output

**Approaches being explored:**
- LLM as controller (OpenAI o3 approach)
- Program synthesis (DreamCoder)
- Neural-symbolic hybrid
- Learned routing networks

---

## 📚 References

- Chollet, F. (2019). "On the Measure of Intelligence"
- Core Knowledge Theory (Spelke, Kinzler)
- ARC-AGI Benchmark: https://arcprize.org/

---

*"Break complex reasoning into learnable skills. Combine them to solve any problem."*
