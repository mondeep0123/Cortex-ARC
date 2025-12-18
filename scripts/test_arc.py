"""Test overall ARC accuracy."""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

from src.brain.visual.solver import Phase3Solver, Task, TrainExample

solver = Phase3Solver()
data_dir = Path('data/arc-agi-1/training')

passed = 0
failed = 0
total = 0

for json_path in data_dir.glob('*.json'):
    with open(json_path) as f:
        data = json.load(f)
    
    train = [TrainExample(
        input_grid=np.array(ex['input'], dtype=np.int8),
        output_grid=np.array(ex['output'], dtype=np.int8)
    ) for ex in data['train']]
    
    test_input = np.array(data['test'][0]['input'], dtype=np.int8)
    test_output = np.array(data['test'][0]['output'], dtype=np.int8)
    
    task = Task(task_id=json_path.stem, train=train, test_input=test_input, test_output=test_output)
    
    is_correct, accuracy = solver.evaluate(task)
    if is_correct:
        passed += 1
    total += 1

print(f"Overall ARC-AGI-1 Accuracy: {passed}/{total} ({100*passed/total:.1f}%)")
