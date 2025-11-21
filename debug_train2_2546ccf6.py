import json
import os.path
import numpy as np
from ArcAgent import rule_complete_quadrant_symmetry

# Load the problem
problem_name = "2546ccf6.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    train_input = np.array(flat_data['train'][1]['input'])
    expected_output = np.array(flat_data['train'][1]['output'])

    result = rule_complete_quadrant_symmetry(train_input)

    diff = (result != expected_output)
    print(f"Matches: {np.sum(result == expected_output)}/{expected_output.size}")
    print(f"Mismatches: {np.sum(diff)}")

    diff_coords = np.argwhere(diff)
    if len(diff_coords) > 0 and len(diff_coords) <= 30:
        print("\nDifferences (row, col) -> Expected vs Got:")
        for r, c in diff_coords:
            print(f"  ({r}, {c}): expected {expected_output[r, c]}, got {result[r, c]}")
