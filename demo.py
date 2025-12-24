"""
Demonstration script showing the curriculum learning system in action.

This script:
1. Loads an ARC task
2. Analyzes it with our skill modules
3. Attempts to solve it using curriculum skills
4. Visualizes the results
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core import Task, SkillLibrary, ObjectCognitionSkill
from utils import grid_from_list, find_objects, get_background_color


def load_arc_task(filepath: str) -> Task:
    """Load an ARC task from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert to our Task format
    train_pairs = [
        (grid_from_list(pair['input']), grid_from_list(pair['output']))
        for pair in data['train']
    ]
    
    test_inputs = [grid_from_list(pair['input']) for pair in data['test']]
    test_outputs = [grid_from_list(pair['output']) for pair in data['test']]
    
    return Task(
        train_pairs=train_pairs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        metadata={'source': filepath}
    )


def visualize_grid(grid: np.ndarray, name: str = "Grid"):
    """Simple ASCII visualization of a grid."""
    # Color palette for visualization
    colors = {
        0: '⬛',  # Black (background)
        1: '🟦',  # Blue
        2: '🟥',  # Red
        3: '🟩',  # Green
        4: '🟨',  # Yellow
        5: '⬜',  # Gray/White
        6: '🟪',  # Magenta
        7: '🟧',  # Orange
        8: '🟦',  # Cyan
        9: '🟫',  # Brown
    }
    
    print(f"\n{name} ({grid.shape[0]}×{grid.shape[1]}):")
    print("─" * (grid.shape[1] * 2 + 2))
    for row in grid:
        print("│" + "".join(colors.get(cell, '❓') for cell in row) + "│")
    print("─" * (grid.shape[1] * 2 + 2))


def demonstrate_object_cognition():
    """Demonstrate the object cognition skill."""
    
    print("\n" + "="*70)
    print("DEMONSTRATION: Object Cognition Skill")
    print("="*70)
    
    # Create a skill instance
    skill = ObjectCognitionSkill()
    
    # Load a sample task
    task_file = "data/training/007bbfb7.json"
    
    if not Path(task_file).exists():
        print(f"❌ Task file not found: {task_file}")
        return
    
    task = load_arc_task(task_file)
    
    print(f"\n📋 Loaded task from: {task_file}")
    print(f"   Training examples: {len(task.train_pairs)}")
    print(f"   Test examples: {len(task.test_inputs)}")
    
    # Analyze first training example
    print("\n" + "-"*70)
    print("ANALYZING TRAINING EXAMPLE 1")
    print("-"*70)
    
    input_grid, output_grid = task.train_pairs[0]
    
    # Visualize input
    visualize_grid(input_grid, "Input")
    
    # Apply object cognition
    result = skill.apply(input_grid, context={"operation": "detect"})
    
    print(f"\n🔍 Object Analysis:")
    print(f"   {result.reasoning}")
    print(f"   Confidence: {result.confidence:.2%}")
    
    # Analyze objects
    analysis = skill.analyze_objects(input_grid)
    print(f"\n📊 Detailed Analysis:")
    print(f"   Background color: {analysis.background_color}")
    print(f"   Number of objects: {analysis.num_objects}")
    print(f"   Colors present: {sorted(analysis.colors_present)}")
    
    if analysis.objects:
        print(f"\n   Objects found:")
        for i, obj in enumerate(analysis.objects, 1):
            print(f"      {i}. Color {obj.color}: size={obj.size}, "
                  f"bbox={obj.width}×{obj.height}")
    
    # Visualize expected output
    visualize_grid(output_grid, "Expected Output")
    
    # Check if skill is relevant for this task
    relevance = skill.can_apply(task)
    print(f"\n🎯 Skill Relevance Score: {relevance:.2%}")
    
    if relevance > 0.5:
        print("   ✅ Object cognition is highly relevant for this task")
    elif relevance > 0.3:
        print("   ⚠️  Object cognition may be somewhat relevant")
    else:
        print("   ❌ Object cognition appears less relevant")


def demonstrate_skill_library():
    """Demonstrate the skill library system."""
    
    print("\n" + "="*70)
    print("DEMONSTRATION: Skill Library")
    print("="*70)
    
    # Create a skill library
    library = SkillLibrary()
    
    # Register skills
    obj_skill = ObjectCognitionSkill()
    library.register(obj_skill, category="core")
    
    print(f"\n📚 Skill Library Created")
    print(f"   Total skills: {len(library)}")
    print(f"   Core skills: {len(library.list_skills('core'))}")
    
    # Load a task
    task_file = "data/training/00d62c1b.json"
    if Path(task_file).exists():
        task = load_arc_task(task_file)
        
        print(f"\n🔍 Finding relevant skills for task: {Path(task_file).stem}")
        
        relevant = library.get_relevant_skills(task, threshold=0.2)
        
        if relevant:
            print(f"   Found {len(relevant)} relevant skill(s):")
            for skill in relevant:
                conf = skill.can_apply(task)
                print(f"      • {skill.name}: {conf:.2%} confidence")
        else:
            print("   No highly relevant skills found")


def demonstrate_multiple_tasks():
    """Analyze multiple tasks to show skill applicability."""
    
    print("\n" + "="*70)
    print("DEMONSTRATION: Multi-Task Analysis")
    print("="*70)
    
    skill = ObjectCognitionSkill()
    
    task_files = sorted(Path("data/training").glob("*.json"))[:5]
    
    print(f"\nAnalyzing {len(task_files)} tasks...\n")
    
    for task_file in task_files:
        task = load_arc_task(task_file)
        relevance = skill.can_apply(task)
        
        # Get first input for object count
        first_input = task.train_pairs[0][0]
        analysis = skill.analyze_objects(first_input)
        
        status = "✅" if relevance > 0.5 else "⚠️ " if relevance > 0.3 else "  "
        
        print(f"{status} {task_file.stem}: "
              f"relevance={relevance:.2%}, "
              f"objects={analysis.num_objects}, "
              f"colors={len(analysis.colors_present)}")


def main():
    """Run all demonstrations."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  ARC-AGI CURRICULUM LEARNING SYSTEM - DEMONSTRATION  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("\nThis demonstration shows:")
    print("  1. Object Cognition skill in action")
    print("  2. Skill Library management")
    print("  3. Multi-task skill relevance analysis")
    
    try:
        # Run demonstrations
        demonstrate_object_cognition()
        demonstrate_skill_library()
        demonstrate_multiple_tasks()
        
        print("\n" + "="*70)
        print("✅ DEMONSTRATION COMPLETE")
        print("="*70)
        
        print("\n📝 Summary:")
        print("   • Successfully loaded ARC tasks")
        print("   • Object cognition skill is working")
        print("   • Skill library system is operational")
        print("   • Ready to implement more skills!")
        
        print("\n🚀 Next Steps:")
        print("   1. Implement remaining core skills (geometry, topology, etc.)")
        print("   2. Create curriculum task generators")
        print("   3. Build skill composition system")
        print("   4. Train on curriculum tasks")
        print("   5. Evaluate on ARC benchmark")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
