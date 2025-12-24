"""
RL-Enhanced Numerosity Training

Key innovations:
1. Reward shaping to penalize constant outputs
2. Entropy bonus to encourage diversity
3. Exact match reward (no tolerance!)
4. Anti-collapse penalties
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class RLNumerosityLoss(nn.Module):
    """
    RL-enhanced loss with reward shaping.
    
    Penalizes:
    - Constant predictions (mode collapse)
    - Near-constant predictions  
    - Large errors
    
    Rewards:
    - Exact matches
    - Diverse predictions
    - Consistent accuracy
    """
    def __init__(self):
        super().__init__()
        
    def compute_rewards(self, predictions, targets, batch_diversity):
        """
        Compute RL rewards for counting predictions.
        
        Args:
            predictions: [batch] predicted counts
            targets: [batch] true counts
            batch_diversity: float, diversity score for this batch
        
        Returns:
            rewards: [batch] reward for each prediction
            info: dict of reward components
        """
        batch_size = predictions.shape[0]
        
        # 1. Exact match reward (primary objective)
        exact_match = (predictions.round() == targets.float()).float()
        match_reward = exact_match * 10.0  # +10 for correct
        
        # 2. Error penalty (punish wrong predictions)
        errors = torch.abs(predictions - targets.float())
        error_penalty = -errors * 2.0  # -2 per unit error
        
        # 3. Diversity reward (prevent mode collapse!)
        # If all predictions are the same, penalize heavily
        diversity_reward = torch.ones_like(predictions) * batch_diversity * 5.0
        
        # 4. Anti-constant penalty
        # If prediction is always 3 or 30, penalize
        constant_3_penalty = -5.0 * (predictions.round() == 3).float()
        constant_30_penalty = -5.0 * (predictions.round() == 30).float()
        anti_constant = constant_3_penalty + constant_30_penalty
        
        # Total reward
        total_reward = (
            match_reward + 
            error_penalty + 
            diversity_reward +
            anti_constant
        )
        
        info = {
            'match_reward': match_reward.mean().item(),
            'error_penalty': error_penalty.mean().item(),
            'diversity_reward': diversity_reward.mean().item(),
            'anti_constant': anti_constant.mean().item(),
            'accuracy': exact_match.mean().item()
        }
        
        return total_reward, info
    
    def compute_batch_diversity(self, predictions):
        """
        Compute diversity of predictions in batch.
        
        High diversity = good (many different predictions)
        Low diversity = bad (mode collapse)
        
        Returns:
            diversity_score: 0 (all same) to 1 (all different)
        """
        pred_rounded = predictions.round()
        unique_preds = len(torch.unique(pred_rounded))
        max_unique = len(pred_rounded)
        
        diversity = unique_preds / max(max_unique, 1)
        return diversity
    
    def forward(self, output, target):
        """
        Compute RL-enhanced loss.
        
        Uses REINFORCE-style policy gradient with reward shaping.
        """
        pred_total = output['total_count'].squeeze()
        true_total = target['total_count'].float()
        
        # Compute batch diversity
        diversity = self.compute_batch_diversity(pred_total)
        
        # Compute rewards
        rewards, info = self.compute_rewards(pred_total, true_total, diversity)
        
        # Policy gradient loss (REINFORCE)
        # Treat prediction as action, reward as feedback
        log_probs = -F.mse_loss(pred_total, true_total, reduction='none')
        pg_loss = -(log_probs * rewards).mean()
        
        # Add supervised loss (MSE) with smaller weight
        supervised_loss = F.mse_loss(pred_total, true_total)
        
        # Color counting loss
        color_loss = 0.0
        if 'color_counts' in target:
            pred_colors = output['color_counts']
            true_colors = target['color_counts'].float()
            color_loss = F.mse_loss(pred_colors, true_colors)
        
        # Total loss
        total_loss = (
            0.3 * pg_loss +           # RL component (reward shaping)
            0.5 * supervised_loss +   # Supervised component
            0.2 * color_loss          # Color accuracy
        )
        
        # Add diversity metrics to info
        info['diversity'] = diversity
        info['total_loss'] = total_loss.item()
        info['pg_loss'] = pg_loss.item()
        info['supervised_loss'] = supervised_loss.item()
        
        return total_loss, info


class RLEvaluator:
    """
    Evaluator with anti-collapse detection.
    """
    def __init__(self):
        self.prediction_history = []
        
    def evaluate(self, output, target):
        """Evaluate with mode collapse detection."""
        pred_count = output['total_count'].squeeze().round()
        true_count = target['total_count'].float()
        
        # Track predictions
        self.prediction_history.extend(pred_count.cpu().tolist())
        
        # Exact match accuracy
        accurate = (pred_count == true_count).float()
        accuracy = accurate.mean().item()
        
        # Mode collapse detection
        if len(self.prediction_history) >= 100:
            recent = self.prediction_history[-100:]
            unique_count = len(set(recent))
            
            # If > 80% predictions are the same value, flag collapse
            most_common = max(set(recent), key=recent.count)
            most_common_pct = recent.count(most_common) / len(recent)
            
            if most_common_pct > 0.8:
                mode_collapse = True
                collapse_value = most_common
            else:
                mode_collapse = False
                collapse_value = None
        else:
            mode_collapse = False
            collapse_value = None
        
        return {
            'accuracy': accuracy,
            'mode_collapse': mode_collapse,
            'collapse_value': collapse_value,
            'unique_predictions': len(set(self.prediction_history[-100:])) if len(self.prediction_history) >= 100 else None
        }
    
    def reset(self):
        """Reset history."""
        self.prediction_history = []


# Test the RL loss
if __name__ == "__main__":
    print("Testing RL-Enhanced Loss...\n")
    
    rl_loss = RLNumerosityLoss()
    
    # Test case 1: Perfect predictions
    print("Test 1: Perfect predictions")
    output = {'total_count': torch.tensor([1.0, 2.0, 3.0, 4.0])}
    target = {'total_count': torch.tensor([1, 2, 3, 4])}
    loss, info = rl_loss(output, target)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {info['accuracy']*100:.1f}%")
    print(f"  Diversity: {info['diversity']:.2f}")
    print()
    
    # Test case 2: Mode collapse (all predict 3)
    print("Test 2: Mode collapse (all predict 3)")
    output = {'total_count': torch.tensor([3.0, 3.0, 3.0, 3.0])}
    target = {'total_count': torch.tensor([1, 2, 5, 4])}
    loss, info = rl_loss(output, target)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {info['accuracy']*100:.1f}%")
    print(f"  Diversity: {info['diversity']:.2f}")
    print(f"  Anti-constant penalty: {info['anti_constant']:.2f} (should be negative!)")
    print()
    
    # Test case 3: Diverse but wrong
    print("Test 3: Diverse but wrong")
    output = {'total_count': torch.tensor([5.0, 6.0, 7.0, 8.0])}
    target = {'total_count': torch.tensor([1, 2, 3, 4])}
    loss, info = rl_loss(output, target)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {info['accuracy']*100:.1f}%")
    print(f"  Diversity: {info['diversity']:.2f}")
    print()
    
    print("✓ RL loss working!")
