# Curriculum Design: Core Cognitive Skills for ARC-AGI

## Curriculum Philosophy

Based on François Chollet's "On the Measure of Intelligence", ARC-AGI tests **fluid intelligence** - the ability to reason and solve novel problems. Our curriculum teaches the **core knowledge priors** that humans naturally develop in early childhood.

We teach **general skills**, not specific puzzle patterns. These skills can be **composed** to solve any reasoning task.

---

## Core Knowledge Priors (Foundation Layer)

These are fundamental concepts that underlie all abstract reasoning:

### 1. Object Cognition
**What it is**: Understanding that the world consists of discrete, persistent objects

**Skills to teach**:
- **Object Permanence**: Objects continue to exist even when occluded
- **Object Cohesion**: Parts of the same object move together
- **Object Boundaries**: Identifying what constitutes a single object vs. multiple objects
- **Object Identity**: Tracking the same object across transformations

**Curriculum tasks**:
- Identify connected components in grids (objects vs. background)
- Track objects as they move or transform
- Distinguish between object parts vs. separate objects
- Recognize when objects overlap vs. merge

---

### 2. Numerosity & Counting
**What it is**: Intuitive understanding of quantity and numerical relationships

**Skills to teach**:
- **Subitizing**: Instantly recognize small quantities (1-4) without counting
- **Counting**: Enumerate objects systematically
- **Comparison**: Determine which is more/less
- **Basic Arithmetic**: Add, subtract, multiply simple quantities

**Curriculum tasks**:
- Count objects of different colors/types
- Compare quantities across different regions
- Create grids with specific counts of elements
- Understand relationships: "same number", "twice as many", etc.

---

### 3. Basic Geometry
**What it is**: Understanding spatial relationships and shapes

**Skills to teach**:
- **Shape Recognition**: Identify basic shapes (square, line, L-shape, etc.)
- **Symmetry**: Detect and create symmetric patterns
- **Orientation**: Understand rotation, flipping, mirroring
- **Size & Scale**: Recognize size relationships and scaling

**Curriculum tasks**:
- Detect lines, rectangles, and other basic shapes
- Identify axes of symmetry (horizontal, vertical, diagonal)
- Apply rotations (90°, 180°, 270°) and reflections
- Scale shapes up or down while preserving proportions

---

### 4. Topology & Containment
**What it is**: Understanding spatial relationships that persist under continuous transformation

**Skills to teach**:
- **Containment**: What is inside vs. outside
- **Connectivity**: What is connected vs. separated
- **Proximity**: What is near vs. far
- **Boundaries**: What marks the edge of a region

**Curriculum tasks**:
- Identify objects inside vs. outside boundaries
- Find connected vs. disconnected regions
- Detect adjacency and neighborhood relationships
- Recognize holes, enclosures, and borders

---

### 5. Elementary Physics (Intuitive)
**What it is**: Basic understanding of how objects behave in space

**Skills to teach**:
- **Gravity/Support**: Objects fall unless supported
- **Contact**: Objects must touch to interact
- **Occlusion**: Nearer objects hide farther ones
- **Persistence of Matter**: Objects don't appear/disappear arbitrarily

**Curriculum tasks**:
- Simulate objects "falling" until supported
- Predict which objects will "roll" or "slide"
- Understand stacking and stability
- Track layers (foreground vs. background)

---

## Cognitive Operations (Skills Layer)

These are the mental operations we apply to the core priors:

### 6. Pattern Recognition & Abstraction
**What it is**: Identifying regularities and extracting essential features

**Skills to teach**:
- **Repetition**: Same element appears multiple times
- **Periodicity**: Regular spacing or timing
- **Hierarchy**: Patterns within patterns (fractals, nested structures)
- **Variation**: Systematic changes (color gradients, size progression)

**Curriculum tasks**:
- Detect repeating motifs in grids
- Identify periodic patterns (every N cells)
- Recognize hierarchical structure (small patterns forming larger ones)
- Find rules governing systematic variations

---

### 7. Transformation & Mapping
**What it is**: Understanding how one configuration becomes another

**Skills to teach**:
- **Translation**: Moving objects in space
- **Rotation & Reflection**: Changing orientation
- **Color Mapping**: Systematic color changes (swap, invert, shift)
- **Composition**: Applying multiple transformations in sequence

**Curriculum tasks**:
- Translate objects by specified offsets
- Rotate/flip objects around different axes
- Learn color substitution rules
- Apply transformation sequences (e.g., "move then rotate")

---

### 8. Analogy & Correspondence
**What it is**: Understanding "same relationship" across different contexts

**Skills to teach**:
- **Structural Analogy**: "A is to B as C is to D"
- **Attribute Mapping**: Corresponding properties (color, size, position)
- **Relational Reasoning**: Same relationship, different objects
- **Proportional Reasoning**: Scaling relationships

**Curriculum tasks**:
- Complete analogy patterns: "If this changed like that, what changes here?"
- Map transformations from one example to novel cases
- Identify corresponding elements across different configurations
- Understand proportional changes

---

### 9. Goal-Directed Reasoning
**What it is**: Working backward from desired outcome to determine actions

**Skills to teach**:
- **Goal Specification**: Identify what the output should achieve
- **Action Selection**: Choose operations that move toward the goal
- **Planning**: Sequence operations to reach the goal
- **Constraint Satisfaction**: Meet multiple requirements simultaneously

**Curriculum tasks**:
- Given output, determine required transformations
- Find shortest path to desired configuration
- Satisfy multiple constraints (e.g., "all reds must be adjacent AND form a square")
- Decompose complex goals into sub-goals

---

### 10. Hypothesis Formation & Testing
**What it is**: Forming theories about rules and verifying them

**Skills to teach**:
- **Rule Induction**: Infer rules from examples
- **Verification**: Check if a rule holds across all cases
- **Revision**: Modify rules when counter-examples found
- **Generalization**: Apply rules to new situations

**Curriculum tasks**:
- Propose transformation rules from input/output pairs
- Test rules on held-out examples
- Refine rules when they fail
- Generalize from 2-3 examples to novel cases

---

## Meta-Cognitive Skills (Control Layer)

These are higher-order skills for managing the reasoning process:

### 11. Attention & Selection
**What it is**: Focusing on relevant information, ignoring distractors

**Skills to teach**:
- **Selective Attention**: Focus on task-relevant features
- **Feature Binding**: Group related attributes together
- **Inhibition**: Ignore irrelevant information
- **Context Sensitivity**: Adjust attention based on task demands

**Curriculum tasks**:
- Identify which grid features are invariant across examples
- Focus on changing vs. unchanging elements
- Filter out irrelevant colors or regions
- Adapt focus based on task structure

---

### 12. Working Memory & Chunking
**What it is**: Holding and manipulating information temporarily

**Skills to teach**:
- **Maintenance**: Keep information active during processing
- **Chunking**: Group elements into meaningful units
- **Integration**: Combine information from multiple sources
- **Update**: Modify held information based on new input

**Curriculum tasks**:
- Remember configurations while applying transformations
- Group related elements to reduce memory load
- Integrate constraints from multiple demonstration pairs
- Update hypotheses as new information arrives

---

### 13. Systematic Search & Exploration
**What it is**: Efficiently exploring the space of possibilities

**Skills to teach**:
- **Breadth-First**: Consider multiple hypotheses in parallel
- **Depth-First**: Fully explore one hypothesis before switching
- **Heuristic Guidance**: Use prior knowledge to guide search
- **Backtracking**: Recognize dead ends and try alternatives

**Curriculum tasks**:
- Generate multiple candidate solutions
- Explore transformations systematically
- Use constraints to prune search space
- Recover from incorrect hypotheses

---

## Curriculum Sequencing Strategy

### Stage 1: Foundations (Weeks 1-2)
Teach core priors in isolation:
1. Object detection and segmentation
2. Basic counting and comparison
3. Shape and symmetry recognition
4. Containment and connectivity

### Stage 2: Basic Operations (Weeks 3-4)
Teach simple transformations:
1. Translation, rotation, reflection
2. Color mapping and substitution
3. Simple pattern repetition
4. One-step transformations

### Stage 3: Composition (Weeks 5-6)
Combine multiple skills:
1. Multi-step transformations
2. Pattern + transformation
3. Constraint satisfaction with 2-3 constraints
4. Simple analogies

### Stage 4: Abstract Reasoning (Weeks 7-8)
Higher-order reasoning:
1. Complex analogies
2. Rule induction from sparse examples
3. Goal-directed planning
4. Hypothesis testing and revision

### Stage 5: Integration (Weeks 9-10)
Full task complexity:
1. Combine all skills flexibly
2. Meta-reasoning about task structure
3. Transfer to novel task types
4. Efficient search and planning

---

## Training Principles

### 1. **Progressive Difficulty**
Start with simple, clear examples. Gradually increase complexity.

### 2. **Minimal Supervision**
Provide sparse feedback. The model should discover patterns, not memorize solutions.

### 3. **Diversity Within Skills**
Each skill should be trained on diverse surface realizations to encourage abstraction.

### 4. **Interleaved Practice**
Mix different skills within training batches to prevent overfitting to specific patterns.

### 5. **Transfer Testing**
Regularly test skills in novel combinations and contexts to ensure true learning.

---

## Success Metrics

- **Skill Acquisition**: Can the model reliably perform each skill in isolation?
- **Skill Composition**: Can the model combine skills to solve multi-step problems?
- **Transfer**: Do skills learned in curriculum tasks transfer to ARC puzzles?
- **Generalization**: Can the model solve ARC puzzles never seen during training?

---

## Next Steps

1. ✅ Define curriculum (this document)
2. 🔄 Analyze ARC-AGI dataset to identify which skills each puzzle requires
3. ⏳ Design synthetic tasks for each curriculum skill
4. ⏳ Build training infrastructure for curriculum learning
5. ⏳ Implement skill-specific evaluation metrics
6. ⏳ Train and iterate on curriculum design

---

## References

- Chollet, F. (2019). "On the Measure of Intelligence" - [arXiv:1911.01547](https://arxiv.org/abs/1911.01547)
- Developmental psychology: Core knowledge systems in infancy
- Cognitive science: Analogical reasoning and abstraction
- AI: Program synthesis and inductive logic programming
