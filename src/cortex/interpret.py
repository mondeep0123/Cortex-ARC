"""
Mechanistic Interpretability for CortexModel

Human-readable explanations of what the model ACTUALLY does.
Logs all puzzle interpretations to file for analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import json


class ModelInterpreter:
    """
    Interprets CortexModel during inference.
    
    Provides human-readable explanations of model behavior.
    """
    
    def __init__(self, model, log_dir: str = "logs/interpretability"):
        self.model = model
        self.attention_maps = []
        self.activations = {}
        self.hooks = []
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log = []
        self.log_file = None
        
    def start_logging(self, session_name: str = None):
        """Start a new logging session."""
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_file = self.log_dir / f"session_{session_name}.txt"
        self.current_log = []
        
        # Write header
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"  MODEL INTERPRETATION LOG\n")
            f.write(f"  Session: {session_name}\n")
            f.write(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
        
        print(f"📝 Logging to: {self.log_file}")
    
    def _log(self, text: str):
        """Add text to current log."""
        self.current_log.append(text)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(text + "\n")
    
    def _register_hooks(self):
        """Register forward hooks to capture internal states."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = []
        self.activations = {}
        
        # Hook attention layers
        for i, layer in enumerate(self.model.reasoning.layers):
            def make_attn_hook(layer_idx):
                def hook(module, input, output):
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
        puzzle_id: str = None,
    ) -> Dict:
        """Run inference with interpretation."""
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
                'puzzle_id': puzzle_id or 'unknown',
                'input': input_grid.cpu().numpy() if isinstance(input_grid, torch.Tensor) else input_grid,
                'prediction': prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction,
                'attention_maps': self.attention_maps,
                'activations': self.activations,
            }
            
            return result
            
        finally:
            self._remove_hooks()
    
    def explain_human(self, result: Dict, verbose: bool = True) -> str:
        """
        Generate a HUMAN-READABLE explanation.
        
        No jargon, just plain English describing what happened.
        """
        lines = []
        puzzle_id = result.get('puzzle_id', 'Unknown')
        
        inp = result['input']
        pred = result['prediction']
        
        # Header
        lines.append(f"\n{'─' * 50}")
        lines.append(f"🧩 PUZZLE: {puzzle_id}")
        lines.append(f"{'─' * 50}")
        
        # Size observation
        lines.append(f"\n📐 SIZE:")
        lines.append(f"   Grid is {inp.shape[0]} rows × {inp.shape[1]} columns")
        if inp.shape != pred.shape:
            lines.append(f"   ⚠️  Output size changed to {pred.shape[0]} × {pred.shape[1]}")
        else:
            lines.append(f"   Output is same size")
        
        # Color observation
        inp_colors = set(inp.flatten()) - {0}
        out_colors = set(pred.flatten()) - {0}
        
        color_names = {
            0: 'black (background)',
            1: 'blue', 2: 'red', 3: 'green', 4: 'yellow',
            5: 'gray', 6: 'magenta', 7: 'orange', 8: 'cyan', 9: 'brown'
        }
        
        lines.append(f"\n🎨 COLORS:")
        inp_color_str = ', '.join([color_names.get(c, str(c)) for c in sorted(inp_colors)])
        out_color_str = ', '.join([color_names.get(c, str(c)) for c in sorted(out_colors)])
        lines.append(f"   Input has: {inp_color_str or 'only background'}")
        lines.append(f"   Output has: {out_color_str or 'only background'}")
        
        # Color changes
        added_colors = out_colors - inp_colors
        removed_colors = inp_colors - out_colors
        if added_colors:
            lines.append(f"   ➕ New colors: {', '.join([color_names.get(c, str(c)) for c in added_colors])}")
        if removed_colors:
            lines.append(f"   ➖ Removed colors: {', '.join([color_names.get(c, str(c)) for c in removed_colors])}")
        
        # What changed
        lines.append(f"\n🔄 WHAT CHANGED:")
        if np.array_equal(inp, pred):
            lines.append(f"   ✅ Nothing! Model output exactly matches input.")
            lines.append(f"   This suggests an IDENTITY transformation.")
        else:
            diff = inp != pred
            changed = diff.sum()
            total = inp.size
            percent = 100 * changed / total
            
            if percent < 10:
                lines.append(f"   🔹 Minor changes: {changed}/{total} pixels ({percent:.1f}%)")
            elif percent < 50:
                lines.append(f"   🔸 Moderate changes: {changed}/{total} pixels ({percent:.1f}%)")
            else:
                lines.append(f"   🔶 Major changes: {changed}/{total} pixels ({percent:.1f}%)")
            
            # Try to identify the transformation type
            if len(out_colors) == 1:
                dom_color = list(out_colors)[0]
                lines.append(f"   Appears to be FILLING with {color_names.get(dom_color, str(dom_color))}")
            elif len(out_colors) < len(inp_colors):
                lines.append(f"   Appears to be COLOR FILTERING (reduced colors)")
            elif out_colors != inp_colors:
                lines.append(f"   Appears to be COLOR TRANSFORMATION (colors changed)")
            else:
                lines.append(f"   Appears to be SPATIAL transformation (colors same, positions changed)")
        
        # Model attention (simplified)
        lines.append(f"\n🧠 MODEL ATTENTION:")
        if result['attention_maps']:
            # Analyze attention focus
            all_max = []
            all_mean = []
            for attn in result['attention_maps']:
                if attn['weights'] is not None:
                    w = attn['weights'][0].mean(dim=0).numpy()
                    all_max.append(w.max())
                    all_mean.append(w.mean())
            
            if all_max:
                avg_max = np.mean(all_max)
                avg_mean = np.mean(all_mean)
                focus_ratio = avg_max / avg_mean if avg_mean > 0 else 1
                
                if focus_ratio > 2:
                    lines.append(f"   🎯 FOCUSED attention - model found specific important areas")
                elif focus_ratio > 1.5:
                    lines.append(f"   🔍 MODERATE focus - model found some patterns")
                else:
                    lines.append(f"   📊 SPREAD attention - model looked at everything equally")
        else:
            lines.append(f"   (Attention data not captured)")
        
        # Confidence assessment
        lines.append(f"\n📊 ASSESSMENT:")
        if np.array_equal(inp, pred) and len(inp_colors) > 0:
            lines.append(f"   Model predicts: Keep input unchanged")
        elif len(out_colors) == 1:
            lines.append(f"   Model predicts: Fill with single color")
        else:
            lines.append(f"   Model predicts: Transform colors/positions")
        
        lines.append("")
        
        explanation = "\n".join(lines)
        
        # Log it
        self._log(explanation)
        
        # Print if verbose
        if verbose:
            print(explanation)
        
        return explanation
    
    def finish_logging(self):
        """Finish the logging session with summary."""
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"  Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  Total puzzles interpreted: {len(self.current_log)}\n")
                f.write("=" * 70 + "\n")
            
            print(f"📁 Log saved to: {self.log_file}")
    
    def interpret_and_explain(
        self,
        input_grid: torch.Tensor,
        puzzle_id: str = None,
        example_inputs: Optional[List[torch.Tensor]] = None,
        example_outputs: Optional[List[torch.Tensor]] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, str]:
        """
        Full pipeline: interpret and explain in human terms.
        
        Returns:
            (prediction, explanation)
        """
        result = self.interpret(input_grid, example_inputs, example_outputs, puzzle_id)
        explanation = self.explain_human(result, verbose=verbose)
        return result['prediction'], explanation


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.cortex.model import CortexModel
    
    print("Testing Human-Readable Interpretability")
    print("=" * 50)
    
    model = CortexModel(embed_dim=128, num_layers=4)
    interpreter = ModelInterpreter(model)
    
    # Start logging
    interpreter.start_logging("test_session")
    
    # Test on a few puzzles
    for i in range(3):
        input_grid = torch.randint(0, 10, (5, 5))
        pred, explanation = interpreter.interpret_and_explain(
            input_grid, 
            puzzle_id=f"test_puzzle_{i+1}"
        )
    
    # Finish logging
    interpreter.finish_logging()
    
    print("\n✓ Human-readable interpretability working!")
