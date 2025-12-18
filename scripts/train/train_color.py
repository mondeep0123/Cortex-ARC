#!/usr/bin/env python3
"""
Train Cortex Model - Phase 1: Color Understanding

Usage:
    python scripts/train/train_color.py --steps 10000 --batch-size 32
"""

import argparse
import sys
sys.path.insert(0, '.')

from src.cortex.model import CortexModel
from src.cortex.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Cortex - Color Phase")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of reasoning layers")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CORTEX MODEL - PHASE 1: COLOR UNDERSTANDING")
    print("=" * 60)
    
    # Create model
    model = CortexModel(
        num_colors=10,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=4,
    )
    
    print(f"\nModel: {model.count_parameters():,} parameters")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Layers: {args.num_layers}")
    print(f"Learning rate: {args.lr}")
    
    # Create trainer
    trainer = Trainer(model, lr=args.lr)
    
    # Resume if checkpoint provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    stats = trainer.train_color_phase(
        num_steps=args.steps,
        batch_size=args.batch_size,
        log_every=100,
        eval_every=500,
        save_every=1000,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final accuracy: {stats['final_accuracy']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
