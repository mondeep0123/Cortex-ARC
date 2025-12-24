"""
ULTIMATE COUNTER - Combining ALL Insights!

1. Object Cognition masks (YOUR insight - eliminates background!)
2. Row-by-row scanning (YOUR counting method!)
3. On-the-fly generation (prevents overfitting!)
4. Simple supervised learning (no complex RL!)

This is THE solution!
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.primitives.distillation_counting import HandcraftedStylePatternGenerator


class UltimateRowByRowCounter(nn.Module):
    """
    The ULTIMATE counter combining all insights:
    - Object-masked input (no background noise!)
    - Row-by-row GRU scanning (human-like!)
    - Running count accumulation
    """
    def __init__(self, d_model=128):
        super().__init__()
        
        self.d_model = d_model
        
        # Input: [pixel_value, mask_value, position]
        self.input_encoder = nn.Linear(3, d_model)
        
        # Row-by-row scanner with larger capacity
        self.row_scanner = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=3,  # Deeper!
            batch_first=True,
            dropout=0.2
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)
        
        # Count decoder
        self.count_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()  # Ensure positive
        )
        
    def forward(self, grid, obj_mask):
        """
        Count using object-masked row-by-row scanning.
        
        Args:
            grid: [batch, h, w] pixel values
            obj_mask: [batch, h, w] object mask (0=background, 1=object)
        
        Returns:
            count: [batch]
        """
        batch_size, h, w = grid.shape
        seq_len = h * w
        
        # Flatten in ROW-MAJOR order (left-to-right, top-to-bottom!)
        grid_flat = grid.view(batch_size, seq_len)  # [batch, seq_len]
        mask_flat = obj_mask.view(batch_size, seq_len)  # [batch, seq_len]
        
        # Position encoding
        positions = torch.arange(seq_len, device=grid.device).unsqueeze(0).expand(batch_size, -1)
        positions_norm = positions.float() / seq_len  # Normalize to [0, 1]
        
        # Combine features: [pixel, mask, position]
        features = torch.stack([
            grid_flat.float() / 9.0,  # Normalize pixels
            mask_flat.float(),        # Object mask
            positions_norm            # Position
        ], dim=-1)  # [batch, seq_len, 3]
        
        # Encode inputs
        encoded = self.input_encoder(features)  # [batch, seq_len, d_model]
        
        # Row-by-row sequential processing
        output, final_hidden = self.row_scanner(encoded)
        # output: [batch, seq_len, d_model] - state at each position
        # final_hidden: [3, batch, d_model] - final accumulated state
        
        # Use final hidden state (represents accumulated count)
        final_state = final_hidden[-1]  # [batch, d_model]
        final_state = self.norm(final_state)
        
        # Decode count
        count = self.count_head(final_state).squeeze(-1)  # [batch]
        
        return count


def train_ultimate_counter(num_epochs=100, samples_per_epoch=1000):
    """
    Train the ULTIMATE counter!
    
    Features:
    - Object masking (eliminates distribution mismatch)
    - Row-by-row scanning (human-like)
    - On-the-fly generation (prevents overfitting)
    - Curriculum learning
    """
    print("="*70)
    print("ULTIMATE COUNTER - COMBINING ALL INSIGHTS!")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Object Cognition masks (no background!)")
    print("  ✓ Row-by-row GRU scanning (like YOU count!)")
    print("  ✓ On-the-fly generation (infinite variety!)")
    print("  ✓ Curriculum learning (easy→hard)")
    print("  ✓ Deep architecture (3-layer GRU)")
    print("\nThis MUST work!\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}, Samples/epoch: {samples_per_epoch}\n")
    
    # Create model
    model = UltimateRowByRowCounter(d_model=128).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters\n")
    
    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    pattern_gen = HandcraftedStylePatternGenerator()
    
    print("="*70)
    print("TRAINING")
    print("="*70)
    print("\nEpoch  Loss     Error    Accuracy  Curriculum         Status")
    print("-"*70)
    
    best_overall = 0.0
    best_easy = 0.0
    best_medium = 0.0
    best_hard = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        
        # Curriculum for training
        if epoch < num_epochs // 4:
            count_range = (0, 5)
            curriculum = "Easy (0-5)"
        elif epoch < num_epochs // 2:
            count_range = (0, 15)
            curriculum = "Medium (0-15)"
        else:
            count_range = (0, 30)
            curriculum = "Hard (0-30)"
        
        epoch_losses = []
        epoch_errors = []
        
        for sample in range(samples_per_epoch):
            # Generate NEW pattern with random seed
            seed = np.random.randint(0, 10000000)
            pattern_gen = HandcraftedStylePatternGenerator(seed=seed)
            grid = pattern_gen.generate(count_range=count_range)
            
            # Ground truth
            target_count = (grid > 0).sum()
            
            # Create object mask (ground truth for now)
            obj_mask = (grid > 0).astype(np.float32)
            
            # To tensors
            grid_tensor = torch.from_numpy(grid).long().unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(obj_mask).float().unsqueeze(0).to(device)
            target_tensor = torch.tensor([target_count], dtype=torch.float, device=device)
            
            # Forward
            pred_count = model(grid_tensor, mask_tensor)
            
            # Loss
            loss = F.mse_loss(pred_count, target_tensor)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track
            error = abs(pred_count.item() - target_count)
            epoch_losses.append(loss.item())
            epoch_errors.append(error)
        
        # EVALUATE ON ALL RANGES (not just current curriculum!)
        model.eval()
        eval_results = {}
        
        with torch.no_grad():
            for eval_name, eval_range in [('easy', (0, 5)), ('medium', (6, 15)), ('hard', (16, 30))]:
                eval_errors = []
                for _ in range(200):  # Evaluate on 200 samples per range
                    seed = np.random.randint(0, 10000000)
                    pattern_gen = HandcraftedStylePatternGenerator(seed=seed)
                    grid = pattern_gen.generate(count_range=eval_range)
                    target_count = (grid > 0).sum()
                    obj_mask = (grid > 0).astype(np.float32)
                    
                    grid_tensor = torch.from_numpy(grid).long().unsqueeze(0).to(device)
                    mask_tensor = torch.from_numpy(obj_mask).float().unsqueeze(0).to(device)
                    
                    pred_count = model(grid_tensor, mask_tensor)
                    error = abs(pred_count.item() - target_count)
                    eval_errors.append(error)
                
                eval_results[eval_name] = np.mean([e < 0.5 for e in eval_errors])
        
        # Overall accuracy (average across ranges)
        overall_acc = np.mean(list(eval_results.values()))
        
        
        # Epoch stats from training
        avg_loss = np.mean(epoch_losses)
        avg_error = np.mean(epoch_errors)
        train_accuracy = np.mean([e < 0.5 for e in epoch_errors])
        
        # Save best models for EACH range
        status_parts = []
        
        if eval_results['easy'] > best_easy:
            best_easy = eval_results['easy']
            torch.save({
                'model_state': model.state_dict(),
                'accuracy': eval_results['easy'],
                'epoch': epoch,
                'range': 'easy (0-5)'
            }, 'checkpoints/ultimate_counter_best_easy.pt')
            status_parts.append("BEST-EASY")
        
        if eval_results['medium'] > best_medium:
            best_medium = eval_results['medium']
            torch.save({
                'model_state': model.state_dict(),
                'accuracy': eval_results['medium'],
                'epoch': epoch,
                'range': 'medium (6-15)'
            }, 'checkpoints/ultimate_counter_best_medium.pt')
            status_parts.append("BEST-MED")
        
        if eval_results['hard'] > best_hard:
            best_hard = eval_results['hard']
            torch.save({
                'model_state': model.state_dict(),
                'accuracy': eval_results['hard'],
                'epoch': epoch,
                'range': 'hard (16-30)'
            }, 'checkpoints/ultimate_counter_best_hard.pt')
            status_parts.append("BEST-HARD")
        
        if overall_acc > best_overall:
            best_overall = overall_acc
            torch.save({
                'model_state': model.state_dict(),
                'accuracy': overall_acc,
                'epoch': epoch,
                'range': 'overall',
                'easy': eval_results['easy'],
                'medium': eval_results['medium'],
                'hard': eval_results['hard']
            }, 'checkpoints/ultimate_counter_best.pt')
            status_parts.append("BEST-OVERALL")
        
        status = " ".join(status_parts) if status_parts else ""
        
        # Print with all metrics
        print(f"{epoch+1:>5}  {avg_loss:>6.4f}   {avg_error:>5.2f}   E:{eval_results['easy']*100:>4.1f}% M:{eval_results['medium']*100:>4.1f}% H:{eval_results['hard']*100:>4.1f}%   {curriculum:>15s}  {status}")
        
        # Early success
        if overall_acc >= 0.95:
            print(f"\n✓✓✓ MASTERED at epoch {epoch+1}!")
            break
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nBest Accuracies:")
    print(f"  Easy (0-5):     {best_easy*100:.2f}%")
    print(f"  Medium (6-15):  {best_medium*100:.2f}%")
    print(f"  Hard (16-30):   {best_hard*100:.2f}%")
    print(f"  Overall:        {best_overall*100:.2f}%")
    
    if best_overall >= 0.80:
        print("\n✓✓✓ BREAKTHROUGH SUCCESS!")
    elif best_overall >= 0.60:
        print("\n✓✓ Major progress!")
    elif best_overall >= 0.40:
        print("\n✓ Good progress!")
    else:
        print("\n⚠ Still learning...")
    
    print("\n✓ Best models saved:")
    print("  - ultimate_counter_best_easy.pt")
    print("  - ultimate_counter_best_medium.pt")
    print("  - ultimate_counter_best_hard.pt")
    print("  - ultimate_counter_best.pt (overall)")
    print("\nNOW TEST EACH MODEL ON HANDCRAFTED BENCHMARK!")


if __name__ == "__main__":
    train_ultimate_counter(num_epochs=100, samples_per_epoch=1000)
