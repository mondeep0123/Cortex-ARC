# ARC-AGI Solver: Curriculum Learning Approach

## Overview
This project implements a **curriculum learning** approach to solve ARC-AGI puzzles by teaching **general cognitive skills** rather than puzzle-specific patterns. The goal is to develop a system that can acquire and apply abstract reasoning abilities to novel tasks.

## Philosophy
Instead of training on specific ARC puzzle solutions, we focus on teaching fundamental cognitive primitives that humans naturally possess. These skills can then be composed to solve any ARC puzzle.

## Project Status
🚧 **Starting from scratch** - Building a clean, principled curriculum-based architecture

## Core Approach
1. **Curriculum Design**: Identify fundamental cognitive skills needed for abstract reasoning
2. **Skill Decomposition**: Break down complex reasoning into teachable primitives  
3. **Progressive Training**: Learn simple skills first, then compose them for complex tasks
4. **Transfer Learning**: Skills learned in one context transfer to novel situations

## Dataset
- **Training Set**: 400 tasks from ARC-AGI-1 (for analysis and validation only)
- **Evaluation Set**: 400 tasks from ARC-AGI-1 (held out for final testing)
- **Curriculum Tasks**: Synthetic tasks designed to teach specific cognitive skills

## Key Insight
> We don't teach the model to solve ARC puzzles. We teach it **how to think** using general-purpose cognitive skills, and puzzle-solving emerges as a consequence.
