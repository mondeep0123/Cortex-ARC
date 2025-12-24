"""
SUBITIZING + ITERATIVE COUNTER - Attempt #21

KEY INSIGHTS:
1. SUBITIZING (0-4): Instant recognition (no counting!)
2. COUNTING (5+): Sequential row-by-row
3. Two-stage training: Perfect row counter → Train adder

YOUR METHOD WITH HUMAN SUBITIZING!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubitizingNetwork(nn.Module):
    """
    SUBITIZING: Instant recognition of 0-4 objects.
    
    Humans don't COUNT small numbers - we just KNOW!
    """
    def __init__(self, d_model=64):
        super().__init__()
        
        # Simple perceptual network (no sequential processing!)
        self.encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        
        # Global pooling (position-invariant!)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Instant recognition
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()
        )
        
        # Initialize for small counts
        nn.init.constant_(self.classifier[-2].bias, 1.5)  # Average 0-4
        
    def forward(self, row, mask):
        """
        Instantly recognize count (no sequential processing!).
        
        Args:
            row: [batch, width]
            mask: [batch, width]
        
        Returns:
            count: [batch] instant recognition!
        """
        batch_size, width = row.shape
        
        # Encode each position
        positions = torch.arange(width, device=row.device).unsqueeze(0).expand(batch_size, -1)
        features = torch.stack([
            row.float() / 9.0,
            mask.float(),
            positions.float() / width
        ], dim=-1)  # [batch, width, 3]
        
        # Encode
        encoded = self.encoder(features)  # [batch, width, d_model]
        
        # Global pooling (position-invariant!)
        pooled = encoded.mean(dim=1)  # [batch, d_model]
        
        # Instant recognition!
        count = self.classifier(pooled).squeeze(-1)  # [batch]
        
        return count


class SequentialRowCounter(nn.Module):
    """Sequential counting for 5+ objects."""
    def __init__(self, d_model=128):
        super().__init__()
        
        self.d_model = d_model
        self.input_encoder = nn.Linear(3, d_model)
        
        self.row_processor = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.count_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()
        )
        
        nn.init.constant_(self.count_head[-2].bias, 7.5)  # Average 5-10
        
    def forward(self, row, mask):
        batch_size, width = row.shape
        
        positions = torch.arange(width, device=row.device).unsqueeze(0).expand(batch_size, -1)
        positions_norm = positions.float() / width
        
        features = torch.stack([
            row.float() / 9.0,
            mask.float(),
            positions_norm
        ], dim=-1)
        
        encoded = self.input_encoder(features)
        output, final_hidden = self.row_processor(encoded)
        count = self.count_head(final_hidden[-1]).squeeze(-1)
        
        return count


class HybridRowCounter(nn.Module):
    """
    HUMAN-LIKE counting:
    - Subitizing (0-4): Instant!
    - Counting (5+): Sequential
    """
    def __init__(self, d_model_sub=64, d_model_seq=128):
        super().__init__()
        
        self.subitizing = SubitizingNetwork(d_model=d_model_sub)
        self.sequential = SequentialRowCounter(d_model=d_model_seq)
        self.threshold = 4.5  # Switch at ~4-5 objects
        
    def forward(self, row, mask):
        """
        Hybrid counting: subitizing for small, sequential for large.
        
        During training, we use both and blend.
        During inference, can use threshold.
        """
        # Get both estimates
        sub_count = self.subitizing(row, mask)
        seq_count = self.sequential(row, mask)
        
        # Soft blend based on estimated count
        # If count is small → trust subitizing
        # If count is large → trust sequential
        blend_weight = torch.sigmoid((sub_count - self.threshold))
        
        # Blend: small counts use subitizing, large use sequential
        final_count = (1 - blend_weight) * sub_count + blend_weight * seq_count
        
        return final_count, sub_count, seq_count


class BinaryAdder(nn.Module):
    """Learn a + b"""
    def __init__(self, d_model=128):
        super().__init__()
        
        self.embed_a = nn.Linear(1, d_model)
        self.embed_b = nn.Linear(1, d_model)
        
        self.adder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()
        )
        
        nn.init.constant_(self.adder[-2].bias, 0.0)
        
    def forward(self, a, b):
        emb_a = self.embed_a(a.unsqueeze(-1))
        emb_b = self.embed_b(b.unsqueeze(-1))
        combined = torch.cat([emb_a, emb_b], dim=-1)
        return self.adder(combined).squeeze(-1)


class SubitizingIterativeCounter(nn.Module):
    """
    COMPLETE HUMAN-LIKE COUNTER:
    1. Subitizing for small counts
    2. Sequential for large counts
    3. Iterative binary addition
    """
    def __init__(self, d_model_sub=64, d_model_seq=128, d_model_add=128):
        super().__init__()
        
        self.row_counter = HybridRowCounter(d_model_sub=d_model_sub, d_model_seq=d_model_seq)
        self.binary_adder = BinaryAdder(d_model=d_model_add)
        
    def forward(self, grid, mask):
        """
        Count with subitizing + iterative addition.
        
        Returns:
            total, row_counts, running_totals, sub_counts, seq_counts
        """
        batch_size, height, width = grid.shape
        device = grid.device
        
        running_total = torch.zeros(batch_size, device=device)
        
        row_counts_list = []
        running_totals_list = []
        sub_counts_list = []
        seq_counts_list = []
        
        for row_idx in range(height):
            row = grid[:, row_idx, :]
            row_mask = mask[:, row_idx, :]
            
            # Hybrid counting (subitizing + sequential)
            row_count, sub_count, seq_count = self.row_counter(row, row_mask)
            
            row_counts_list.append(row_count)
            sub_counts_list.append(sub_count)
            seq_counts_list.append(seq_count)
            
            # Binary addition
            running_total = self.binary_adder(running_total, row_count)
            running_totals_list.append(running_total)
        
        row_counts = torch.stack(row_counts_list, dim=1)
        running_totals = torch.stack(running_totals_list, dim=1)
        sub_counts = torch.stack(sub_counts_list, dim=1)
        seq_counts = torch.stack(seq_counts_list, dim=1)
        
        return running_total, row_counts, running_totals, sub_counts, seq_counts


# Test
if __name__ == "__main__":
    print("Testing SUBITIZING + ITERATIVE Counter...\n")
    print("HUMAN-LIKE COUNTING:")
    print("  ✓ Subitizing (0-4): Instant recognition!")
    print("  ✓ Counting (5+): Sequential")
    print("  ✓ Binary addition: a+b\n")
    
    model = SubitizingIterativeCounter(d_model_sub=64, d_model_seq=128, d_model_add=128)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    
    # Test
    test_grid = torch.randint(0, 10, (2, 5, 6))
    test_mask = (test_grid > 0).float()
    
    print(f"Input: {test_grid.shape}")
    print(f"Sample grid:\n{test_grid[0]}\n")
    
    total, row_counts, running_totals, sub_counts, seq_counts = model(test_grid, test_mask)
    
    print(f"Sample predictions:")
    print(f"  Subitizing: {[round(c.item(), 2) for c in sub_counts[0]]}")
    print(f"  Sequential: {[round(c.item(), 2) for c in seq_counts[0]]}")
    print(f"  Final row counts: {[round(c.item(), 2) for c in row_counts[0]]}")
    print(f"  Running totals: {[round(t.item(), 2) for t in running_totals[0]]}")
    print(f"  Final total: {total[0].item():.2f}")
    print(f"  Actual: {(test_grid[0] > 0).sum().item()}\n")
    
    print("✓ Subitizing + Iterative counter ready!")
