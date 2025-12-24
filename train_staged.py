"""
STAGED TRAINING - Attempt #29

SAME MODEL, DIFFERENT DATA PER STAGE:

Stage 1: Subitizing (chunks from grids → count 0-4)
Stage 2: Adder on PURE NUMBERS (exhaustive a+b=c pairs!)
Stage 3: Combine and test

This should finally work!
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from primitives.distillation_counting import HandcraftedStylePatternGenerator


class SubitizingModule(nn.Module):
    """Counts non-zero pixels in chunk (0-4)."""
    def __init__(self, d_model=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()
        )
        
        nn.init.constant_(self.classifier[-2].bias, 2.0)
        
    def forward(self, chunk):
        batch_size, chunk_size = chunk.shape
        
        # Features: is_nonzero, position
        is_nonzero = (chunk > 0).float()
        positions = torch.arange(chunk_size, device=chunk.device).unsqueeze(0).expand(batch_size, -1)
        
        features = torch.stack([
            is_nonzero,
            positions.float() / chunk_size
        ], dim=-1)
        
        encoded = self.encoder(features)
        pooled = encoded.mean(dim=1)
        count = self.classifier(pooled).squeeze(-1)
        
        return count


class PureArithmeticAdder(nn.Module):
    """
    Adder trained on PURE NUMBERS!
    
    Input: (a, b) where a, b are integers
    Output: a + b
    
    Trained exhaustively on ALL pairs!
    """
    def __init__(self, hidden_size=128, max_value=50):
        super().__init__()
        
        self.max_value = max_value
        
        # Simple MLP that learns addition
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize close to identity for addition
        nn.init.constant_(self.network[-1].bias, 5.0)
        
    def forward(self, a, b):
        """
        a, b: [batch] integers
        Returns: [batch] predicted sum
        """
        # Normalize inputs
        a_norm = a / self.max_value
        b_norm = b / self.max_value
        
        x = torch.stack([a_norm, b_norm], dim=-1)
        
        # Output in normalized space, then scale back
        out_norm = self.network(x).squeeze(-1)
        out = out_norm * self.max_value  # Scale back
        
        return out


class StagedCounter(nn.Module):
    """Counter with staged training."""
    def __init__(self, d_model_sub=64, hidden_size=128, chunk_size=4):
        super().__init__()
        
        self.subitizing = SubitizingModule(d_model=d_model_sub)
        self.adder = PureArithmeticAdder(hidden_size=hidden_size, max_value=50)
        self.chunk_size = chunk_size
        
    def extract_objects(self, row, mask):
        batch_size = row.shape[0]
        extracted_list = []
        max_objects = 0
        
        for b in range(batch_size):
            non_zero_mask = mask[b] > 0
            objects = row[b][non_zero_mask]
            extracted_list.append(objects)
            max_objects = max(max_objects, len(objects))
        
        extracted = torch.zeros(batch_size, max(max_objects, 1), device=row.device, dtype=row.dtype)
        for b in range(batch_size):
            if len(extracted_list[b]) > 0:
                extracted[b, :len(extracted_list[b])] = extracted_list[b]
        
        return extracted
        
    def forward(self, grid, mask):
        batch_size, height, width = grid.shape
        device = grid.device
        
        running_total = torch.zeros(batch_size, device=device)
        
        for row_idx in range(height):
            row = grid[:, row_idx, :]
            row_mask = mask[:, row_idx, :]
            
            # Check if row has any objects
            if row_mask.sum() == 0:
                continue  # Skip empty rows!
            
            extracted = self.extract_objects(row, row_mask)
            
            # Double-check: if no actual objects, skip
            if extracted.abs().sum() == 0:
                continue
            
            num_objects = int(row_mask.sum().item())  # Actual count of objects!
            num_chunks = (num_objects + self.chunk_size - 1) // self.chunk_size
            
            for chunk_idx in range(num_chunks):
                start = chunk_idx * self.chunk_size
                end = min(start + self.chunk_size, num_objects)
                
                chunk = extracted[:, start:end]
                if chunk.shape[1] < self.chunk_size:
                    chunk = F.pad(chunk, (0, self.chunk_size - chunk.shape[1]), value=0)
                
                # Subitize: count non-zero in chunk
                chunk_count = torch.round(self.subitizing(chunk))
                
                # Add using pre-trained adder
                running_total = torch.round(self.adder(running_total, chunk_count))
        
        return running_total


def train_staged():
    print("="*70)
    print("STAGED TRAINING - ATTEMPT #29")
    print("="*70)
    print("\nSAME MODEL, DIFFERENT DATA:")
    print("  Stage 1: Subitizing on chunks")
    print("  Stage 2: Adder on PURE NUMBERS (exhaustive!)")
    print("  Stage 3: Test combined\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    model = StagedCounter(d_model_sub=64, hidden_size=128, chunk_size=4).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters\n")
    
    # ==================== STAGE 1: SUBITIZING ====================
    print("="*70)
    print("STAGE 1: TRAINING SUBITIZING (on chunks)")
    print("="*70)
    
    sub_params = list(model.subitizing.parameters())
    optimizer_sub = torch.optim.Adam(sub_params, lr=0.001)
    
    for epoch in range(30):
        model.train()
        losses = []
        
        for _ in range(500):
            seed = np.random.randint(0, 10000000)
            pattern_gen = HandcraftedStylePatternGenerator(seed=seed)
            grid = pattern_gen.generate(count_range=(0, 30))
            
            for row in grid:
                extracted = row[row > 0]
                if len(extracted) < 1:
                    continue
                
                for i in range(0, len(extracted), 4):
                    chunk = extracted[i:i+4]
                    true_count = len(chunk)  # Count of non-zero!
                    
                    chunk_padded = torch.zeros(1, 4, device=device)
                    chunk_padded[0, :len(chunk)] = torch.tensor(chunk, dtype=torch.float, device=device)
                    
                    pred = model.subitizing(chunk_padded)
                    loss = (pred - true_count) ** 2
                    
                    optimizer_sub.zero_grad()
                    loss.backward()
                    optimizer_sub.step()
                    
                    losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {np.mean(losses):.4f}")
    
    # Freeze subitizing
    for param in model.subitizing.parameters():
        param.requires_grad = False
    
    print("\n✓ Subitizing trained and FROZEN!")
    
    # ==================== STAGE 2: PURE ARITHMETIC ====================
    print(f"\n{'='*70}")
    print("STAGE 2: TRAINING ADDER (on PURE NUMBERS)")
    print("="*70)
    print("\nExhaustive training: all pairs (a,b) where a,b in [0,30]")
    print("Total pairs: 31 * 31 = 961\n")
    
    adder_params = list(model.adder.parameters())
    optimizer_adder = torch.optim.Adam(adder_params, lr=0.01)
    
    # Generate ALL pairs exhaustively
    all_pairs = []
    for a in range(0, 31):
        for b in range(0, 31):
            all_pairs.append((a, b, a + b))
    
    print(f"Training on {len(all_pairs)} exhaustive pairs...")
    
    for epoch in range(100):
        model.train()
        np.random.shuffle(all_pairs)
        
        epoch_loss = 0
        for (a, b, target) in all_pairs:
            a_t = torch.tensor([float(a)], device=device)
            b_t = torch.tensor([float(b)], device=device)
            target_t = torch.tensor([float(target)], device=device)
            
            pred = model.adder(a_t, b_t)
            loss = (pred - target_t) ** 2
            
            optimizer_adder.zero_grad()
            loss.backward()
            optimizer_adder.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(all_pairs)
        
        # Test accuracy
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for (a, b, target) in all_pairs:
                    a_t = torch.tensor([float(a)], device=device)
                    b_t = torch.tensor([float(b)], device=device)
                    pred = model.adder(a_t, b_t)
                    if abs(round(pred.item()) - target) < 0.5:
                        correct += 1
            
            acc = correct / len(all_pairs) * 100
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.1f}%")
            
            if acc >= 99.0:
                print("\n✓✓✓ ADDER MASTERED ARITHMETIC!")
                break
    
    # Freeze adder too
    for param in model.adder.parameters():
        param.requires_grad = False
    
    print("\n✓ Adder trained and FROZEN!")
    
    # ==================== STAGE 3: TEST COMBINED ====================
    print(f"\n{'='*70}")
    print("STAGE 3: TESTING COMBINED MODEL ON GRIDS")
    print("="*70)
    
    model.eval()
    
    for eval_name, eval_range in [('Easy (0-5)', (0, 5)), ('Medium (6-15)', (6, 15)), ('Hard (16-30)', (16, 30))]:
        correct = 0
        total = 100
        
        with torch.no_grad():
            for _ in range(total):
                seed = np.random.randint(0, 10000000)
                pattern_gen = HandcraftedStylePatternGenerator(seed=seed)
                grid = pattern_gen.generate(count_range=eval_range)
                target_count = (grid > 0).sum()
                
                grid_tensor = torch.from_numpy(grid).long().unsqueeze(0).to(device)
                obj_mask = (grid_tensor > 0).float()
                
                pred_count = model(grid_tensor, obj_mask)
                
                if abs(pred_count.item() - target_count) < 0.5:
                    correct += 1
        
        print(f"  {eval_name}: {correct}/{total} = {correct}%")
    
    # Save
    torch.save({'model_state': model.state_dict()}, 'checkpoints/staged_model.pt')
    
    print(f"\n{'='*70}")
    print("STAGED TRAINING COMPLETE")
    print("="*70)
    print("\n✓ Model saved!")
    print("\nNOW TEST ON HANDCRAFTED BENCHMARK!")


if __name__ == "__main__":
    train_staged()
