"""
BREAKTHROUGH ARCHITECTURE: Neural Accumulator Counter

Key insight: Teach the model to COUNT ALGORITHMICALLY!

Components:
1. Object Detection (from Object Cognition)
2. Sequential Iterator (process objects one-by-one)
3. Neural Accumulator (learns increment operation)
4. NALU (Neural Arithmetic Logic Unit) for addition

This learns the COUNTING ALGORITHM, not pattern matching!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.primitives.base_primitive import BasePrimitiveModel, PrimitiveSkillConfig


class NALU(nn.Module):
    """
    Neural Arithmetic Logic Unit (DeepMind 2018)
    
    Learns addition, subtraction, multiplication, division.
    Perfect for COUNT = COUNT + 1 operation!
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # NAC (Neural Accumulator) - learns addition
        self.W_hat = nn.Parameter(torch.Tensor(out_features, in_features))
        self.M_hat = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Gate for selecting operation
        self.G = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_hat)
        nn.init.kaiming_uniform_(self.M_hat)
        nn.init.kaiming_uniform_(self.G)
        
    def forward(self, x):
        # NAC: W = tanh(W_hat) * sigmoid(M_hat)
        # This constrains W to be close to {-1, 0, 1}
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        
        # Addition/subtraction operation
        a = F.linear(x, W, self.bias)
        
        # Gate (for now, just use addition)
        return a


class NeuralAccumulator(nn.Module):
    """
    Accumulator that learns to INCREMENT a counter.
    
    Implements: count_new = count_old + 1
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # NALU for learning increment
        self.increment_unit = NALU(hidden_dim + 1, 1)  # [state, 1] -> [count]
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, current_count, object_feature):
        """
        Update count based on object presence.
        
        Args:
            current_count: [batch, 1] current count
            object_feature: [batch, hidden_dim] object representation
        
        Returns:
            new_count: [batch, 1] incremented count
        """
        # Encode object state
        state = self.state_encoder(object_feature)  # [batch, hidden_dim]
        
        # Concatenate with current count
        input_vec = torch.cat([state, current_count], dim=1)  # [batch, hidden_dim+1]
        
        # Learn increment via NALU
        increment = self.increment_unit(input_vec)  # [batch, 1]
        
        # New count
        new_count = current_count + F.relu(increment)  # Ensure positive increment
        
        return new_count


class SequentialObjectProcessor(nn.Module):
    """
    Process objects SEQUENTIALLY like humans count.
    
    Not parallel! One object at a time.
    """
    def __init__(self, feature_dim=128, hidden_dim=64):
        super().__init__()
        
        # Extract individual object features
        self.object_encoder = nn.Sequential(
            nn.Conv2d(128, 64, 1),  # 1x1 conv
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 1)
        )
        
        # Object existence detector
        self.existence_gate = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, object_mask):
        """
        Extract individual object representations.
        
        Args:
            features: [batch, 128, h, w]
            object_mask: [batch, 1, h, w]
        
        Returns:
            object_features: [batch, max_objects, hidden_dim]
            object_exists: [batch, max_objects] - 0 or 1
        """
        batch, _, h, w = features.shape
        
        # Encode objects
        obj_features = self.object_encoder(features)  # [batch, hidden_dim, h, w]
        
        # Find object positions (non-zero in mask)
        # For each position, extract feature if object exists
        obj_features_flat = obj_features.permute(0, 2, 3, 1).reshape(batch, h*w, -1)  # [batch, h*w, hidden_dim]
        object_mask_flat = object_mask.squeeze(1).reshape(batch, h*w)  # [batch, h*w]
        
        # Select top-k positions with highest mask values
        max_objects = 30
        topk_values, topk_indices = torch.topk(object_mask_flat, min(max_objects, h*w), dim=1)
        
        # Gather features at these positions
        batch_indices = torch.arange(batch).unsqueeze(1).expand(-1, topk_indices.size(1))
        selected_features = obj_features_flat[batch_indices, topk_indices]  # [batch, max_objects, hidden_dim]
        
        # Existence: 1 if mask > 0.5, else 0
        object_exists = (topk_values > 0.5).float()  # [batch, max_objects]
        
        return selected_features, object_exists


class NumerosityPrimitive(BasePrimitiveModel):
    """
    BREAKTHROUGH: Neural Accumulator Counter
    
    Learns the counting ALGORITHM, not pattern matching!
    """
    
    def __init__(self, config: PrimitiveSkillConfig):
        super().__init__(config)
        
        # Object Cognition
        self.obj_cog_model = None
        
        # Feature extraction
        self.color_embed = nn.Embedding(10, 64)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(66, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Sequential object processor
        self.object_processor = SequentialObjectProcessor(
            feature_dim=128,
            hidden_dim=64
        )
        
        # Neural accumulator (learns to count!)
        self.accumulator = NeuralAccumulator(hidden_dim=64)
        
        # Per-color counting
        self.color_counter = nn.Linear(128, 10)
        
    def load_object_cognition(self, checkpoint_path):
        """Load Object Cognition."""
        from src.primitives.object_cognition_primitive import ObjectCognitionPrimitive
        import src.primitives.base_primitive
        
        sys.modules['primitives.base_primitive'] = src.primitives.base_primitive
        sys.modules['primitives.object_cognition_primitive'] = sys.modules['src.primitives.object_cognition_primitive']
        
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        obj_config = checkpoint.get('config', self.config)
        obj_config.device = self.config.device
        
        self.obj_cog_model = ObjectCognitionPrimitive(obj_config).to(self.config.device)
        self.obj_cog_model.load_state_dict(checkpoint['model_state'])
        self.obj_cog_model.eval()
        
        for param in self.obj_cog_model.parameters():
            param.requires_grad = False
        
        print(f"✓ Loaded Object Cognition")
        
    def forward(self, grid_tensor):
        """
        Sequential counting with neural accumulator!
        """
        batch_size, h, w = grid_tensor.shape
        
        # Get object mask
        if self.obj_cog_model is not None:
            with torch.no_grad():
                obj_output = self.obj_cog_model(grid_tensor)
                obj_seg = obj_output['segmentation']
                obj_boundary = obj_output['boundaries']
        else:
            obj_seg = (grid_tensor > 0).unsqueeze(1).float()
            obj_boundary = torch.zeros_like(obj_seg)
        
        # Feature extraction
        color_features = self.color_embed(grid_tensor.long()).permute(0, 3, 1, 2)
        fused = torch.cat([color_features, obj_seg, obj_boundary], dim=1)
        features = self.feature_fusion(fused)  # [batch, 128, h, w]
        
        # Extract individual objects
        object_features, object_exists = self.object_processor(features, obj_seg)
        # [batch, max_objects, 64], [batch, max_objects]
        
        # SEQUENTIAL COUNTING (like humans!)
        count = torch.zeros(batch_size, 1, device=grid_tensor.device)
        
        for i in range(object_features.size(1)):
            obj_feat = object_features[:, i, :]  # [batch, 64]
            obj_exist = object_exists[:, i: i+1]  # [batch, 1]
            
            # If object exists, increment count
            count = count + obj_exist * F.relu(self.accumulator(count, obj_feat))
        
        # Per-color counts
        global_feat = features.mean(dim=(2, 3))
        color_counts = self.color_counter(global_feat)
        color_counts = F.relu(color_counts)
        
        return {
            'total_count': count,
            'color_counts': color_counts,
            'max_color_logits': color_counts
        }
    
    def compute_loss(self, output, target):
        """Simple supervised loss - let NALU learn arithmetic!"""
        loss = 0.0
        
        pred_total = output['total_count'].squeeze()
        true_total = target['total_count'].float()
        loss += 100.0 * F.mse_loss(pred_total, true_total)  # High weight!
        
        if 'color_counts' in target:
            pred_colors = output['color_counts']
            true_colors = target['color_counts'].float()
            loss += 10.0 * F.mse_loss(pred_colors, true_colors)
        
        return loss
    
    def evaluate_output(self, output, target):
        """Exact match."""
        pred_count = output['total_count'].squeeze().round()
        true_count = target['total_count'].float()
        return (pred_count == true_count).float().mean().item()
    
    def predict(self, grid: np.ndarray) -> dict:
        """Prediction."""
        self.eval()
        with torch.no_grad():
            grid_tensor = torch.from_numpy(grid).long().unsqueeze(0).to(self.config.device)
            output = self(grid_tensor)
            
            total = int(round(output['total_count'].item()))
            total = max(0, min(30, total))
            
            colors = output['color_counts'].squeeze().round().long().cpu().numpy()
            colors = np.clip(colors, 0, 30)
            
            max_color = output['max_color_logits'].argmax(dim=1).item()
            
            return {
                'total_count': total,
                'color_counts': colors.tolist(),
                'max_color': max_color
            }
