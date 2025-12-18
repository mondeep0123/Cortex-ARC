"""Test only consistent-position CROP puzzles."""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

from src.brain.visual.solver import Phase3Solver, Task, TrainExample
from src.brain.visual.reasoning import detect_grid_transform_with_params, TransformType

data_dir = Path('data/arc-agi-1/training')
solver = Phase3Solver()

consistent_pos_puzzles = []

for json_path in data_dir.glob('*.json'):
    with open(json_path) as f:
        data = json.load(f)
    
    all_crop = True
    crop_positions = []
    
    for ex in data['train']:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        t, params = detect_grid_transform_with_params(inp, out)
        if t != TransformType.CROP:
            all_crop = False
            break
        crop_positions.append((params.get('row'), params.get('col'), params.get('height'), params.get('width')))
    
    if all_crop and crop_positions:
        positions = [(p[0], p[1]) for p in crop_positions]
        if len(set(positions)) == 1:
            consistent_pos_puzzles.append((json_path, data))

print(f"Testing {len(consistent_pos_puzzles)} consistent-position CROP puzzles...")
print("=" * 60)

passed = 0
failed = 0

for json_path, data in consistent_pos_puzzles:
    train = [TrainExample(
        input_grid=np.array(ex['input'], dtype=np.int8),
        output_grid=np.array(ex['output'], dtype=np.int8)
    ) for ex in data['train']]
    
    test_input = np.array(data['test'][0]['input'], dtype=np.int8)
    test_output = np.array(data['test'][0]['output'], dtype=np.int8)
    
    task = Task(task_id=json_path.stem, train=train, test_input=test_input, test_output=test_output)
    
    is_correct, accuracy = solver.evaluate(task)
    
    status = "PASS" if is_correct else "FAIL"
    print(f"  {status}: {json_path.stem} ({accuracy:.1%})")
    
    if is_correct:
        passed += 1
    else:
        failed += 1
        pred = solver.solve(task)
        print(f"    Expected: {test_output.shape}, Got: {pred.shape}")

print("=" * 60)
print(f"Result: {passed}/{passed+failed}")
