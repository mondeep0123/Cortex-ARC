# Contributing to Cortex-ARC

First off, thank you for considering contributing to Cortex-ARC! 🎉

## 🚀 How Can I Contribute?

### 1. Reporting Bugs

- Check if the bug has already been reported in [Issues](https://github.com/mondeep0123/Cortex-ARC/issues)
- If not, create a new issue with:
  - Clear title
  - Steps to reproduce
  - Expected vs actual behavior
  - Your environment (Python version, OS, etc.)

### 2. Suggesting Features

- Open an issue with the tag `enhancement`
- Describe the feature and why it would be useful
- If possible, provide examples or references

### 3. Contributing Code

#### High-Priority Areas:
- **New Primitives**: Geometry, Topology, Symmetry detection
- **Color-Aware Masking**: Count specific colors
- **Meta-Controller**: Learn which primitives to use when
- **More Benchmark Puzzles**: Expand the handcrafted test set

#### Steps:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test your changes
5. Commit with clear messages: `git commit -m "Add: New symmetry primitive"`
6. Push: `git push origin feature/your-feature`
7. Open a Pull Request

### 4. Improving Documentation

- Fix typos
- Add examples
- Clarify explanations
- Add diagrams

## 📋 Code Style

- Use clear, descriptive variable names
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and small

## 🧪 Testing

Before submitting:
```bash
python train_staged.py  # Should reach 100% on benchmark
```

## 💬 Questions?

- Open an issue with the `question` tag
- Discord: [mondeep.blend](https://discord.com/users/1085083654251872357)

## 🎯 Project Philosophy

> "One model. All cognitive abilities. Any domain."

We believe in:
- **Compositional skills** over end-to-end black boxes
- **Human-like reasoning** (subitizing, chunking, etc.)
- **Small, efficient models** when possible
- **Reproducible results** with proper benchmarks

---

*"I am no expert, a vibecoder but came this far. I won't surrender!"*

Thank you for helping make Cortex-ARC better! 🙏
