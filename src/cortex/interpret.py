"""
Mechanistic Interpretability for CortexModel

See what the model ACTUALLY does, not what it would SAY it does.

Methods:
1. Attention visualization - Where does model look?
2. Activation analysis - What features are active?
3. Attribution - What input affects output?
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class ModelInterpreter:
    """
    Interprets CortexModel during inference.
    
    Captures internal computations to understand reasoning.
    """
    
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self.activations = {}
        self.hooks = []
        
    def _register_hooks(self):
        """Register forward hooks to capture internal states."""
        
        # Clear previous hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = []
        self.activations = {}
        
        # Hook attention layers
        for i, layer in enumerate(self.model.reasoning.layers):
            def make_attn_hook(layer_idx):
                def hook(module, input, output):
                    # output[1] is attention weights
                    if isinstance(output, tuple) and len(output) > 1:
                        self.attention_maps.append({
                            'layer': layer_idx,
                            'weights': output[1].detach().cpu() if output[1] is not None else None
                        })
                return hook
            
            h = layer.attn.register_forward_hook(make_attn_hook(i))
            self.hooks.append(h)
        
        # Hook color embeddings
        def color_hook(module, input, output):
            self.activations['color_embed'] = output.detach().cpu()
        h = self.model.encoder.color_embed.register_forward_hook(color_hook)
        self.hooks.append(h)
        
        # Hook final reasoning output
        def reasoning_hook(module, input, output):
            self.activations['reasoning_out'] = output.detach().cpu()
        h = self.model.reasoning.register_forward_hook(reasoning_hook)
        self.hooks.append(h)
    
    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def interpret(
        self, 
        input_grid: torch.Tensor,
        example_inputs: Optional[List[torch.Tensor]] = None,
        example_outputs: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """
        Run inference with interpretation.
        
        Args:
            input_grid: Input grid [H, W]
            example_inputs: Optional few-shot examples
            example_outputs: Optional few-shot examples
            
        Returns:
            Dict with prediction and interpretation data
        """
        self._register_hooks()
        
        try:
            self.model.eval()
            with torch.no_grad():
                prediction = self.model.predict(
                    input_grid, 
                    example_inputs, 
                    example_outputs
                )
            
            result = {
                'input': input_grid.cpu().numpy() if isinstance(input_grid, torch.Tensor) else input_grid,
                'prediction': prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction,
                'attention_maps': self.attention_maps,
                'activations': self.activations,
            }
            
            return result
            
        finally:
            self._remove_hooks()
    
    def visualize_attention(
        self, 
        result: Dict, 
        layer: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention patterns.
        
        Shows which positions the model relates to which.
        """
        if not result['attention_maps']:
            print("No attention data captured (model may not return attention weights)")
            return
        
        attn_data = result['attention_maps'][layer]
        weights = attn_data['weights']
        
        if weights is None:
            print("Attention weights not available")
            return
        
        # weights: [batch, heads, seq_len, seq_len]
        # Average over heads
        avg_attn = weights[0].mean(dim=0).numpy()  # [seq_len, seq_len]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_attn, cmap='hot')
        plt.colorbar(label='Attention weight')
        plt.xlabel('Key position')
        plt.ylabel('Query position')
        plt.title(f'Layer {layer} Attention Pattern')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def visualize_input_output(self, result: Dict, save_path: Optional[str] = None):
        """Visualize input and predicted output side by side."""
        
        inp = result['input']
        pred = result['prediction']
        
        # ARC color palette
        colors = [
            '#000000',  # 0: black
            '#0074D9',  # 1: blue
            '#FF4136',  # 2: red
            '#2ECC40',  # 3: green
            '#FFDC00',  # 4: yellow
            '#AAAAAA',  # 5: gray
            '#F012BE',  # 6: magenta
            '#FF851B',  # 7: orange
            '#7FDBFF',  # 8: cyan
            '#870C25',  # 9: brown
        ]
        cmap = mcolors.ListedColormap(colors)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(inp, cmap=cmap, vmin=0, vmax=9)
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        axes[1].imshow(pred, cmap=cmap, vmin=0, vmax=9)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def get_saliency(self, input_grid: torch.Tensor, target_pos: Tuple[int, int]) -> np.ndarray:
        """
        Compute saliency: which input pixels affect output at target_pos.
        
        Uses gradient-based attribution.
        """
        input_grid = input_grid.clone().requires_grad_(True).float()
        
        self.model.eval()
        
        # Forward pass
        logits, _ = self.model(input_grid.unsqueeze(0).long())
        
        # Get logit at target position
        target_logit = logits[0, target_pos[0], target_pos[1], :].sum()
        
        # Backward to get gradients
        target_logit.backward()
        
        # Gradient magnitude as saliency
        saliency = input_grid.grad.abs().numpy()
        
        return saliency
    
    def explain_prediction(self, result: Dict) -> str:
        """
        Generate a text explanation of what we observed.
        
        This is NOT what model "thinks" - it's our interpretation
        of observed internal states.
        """
        lines = []
        lines.append("=== Interpretation Report ===\n")
        
        inp = result['input']
        pred = result['prediction']
        
        # Size analysis
        lines.append(f"Input size: {inp.shape}")
        lines.append(f"Output size: {pred.shape}")
        
        # Color analysis
        inp_colors = set(inp.flatten()) - {0}
        out_colors = set(pred.flatten()) - {0}
        
        lines.append(f"\nInput colors: {sorted(inp_colors)}")
        lines.append(f"Output colors: {sorted(out_colors)}")
        
        # Changes
        if np.array_equal(inp, pred):
            lines.append("\nTransformation: IDENTITY (no change)")
        else:
            diff = inp != pred
            changed_count = diff.sum()
            lines.append(f"\nPixels changed: {changed_count} / {inp.size}")
        
        # Attention analysis
        if result['attention_maps']:
            lines.append(f"\nAttention layers captured: {len(result['attention_maps'])}")
            for attn in result['attention_maps']:
                if attn['weights'] is not None:
                    # Check if attention is focused or spread
                    weights = attn['weights'][0].mean(dim=0).numpy()
                    max_attn = weights.max()
                    mean_attn = weights.mean()
                    lines.append(f"  Layer {attn['layer']}: max={max_attn:.3f}, mean={mean_attn:.3f}")
        
        return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.cortex.model import CortexModel
    
    print("Testing ModelInterpreter...")
    
    model = CortexModel(embed_dim=128, num_layers=4)
    interpreter = ModelInterpreter(model)
    
    # Test input
    input_grid = torch.randint(0, 10, (5, 5))
    
    # Interpret
    result = interpreter.interpret(input_grid)
    
    print(interpreter.explain_prediction(result))
    print("\n✓ ModelInterpreter working!")
