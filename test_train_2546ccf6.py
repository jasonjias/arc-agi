import json
import os.path
import numpy as np
from ArcAgent import rule_complete_quadrant_symmetry

# Load the problem
problem_name = "2546ccf6.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    for i, example in enumerate(flat_data['train']):
        train_input = np.array(example['input'])
        expected_output = np.array(example['output'])

        result = rule_complete_quadrant_symmetry(train_input)

        matches = np.array_equal(result, expected_output)
        if result.shape == expected_output.shape:
            match_pct = np.sum(result == expected_output) / expected_output.size * 100
            print(f"Training example {i+1}: {match_pct:.1f}% match - {'PASS' if matches else 'FAIL'}")
        else:
            print(f"Training example {i+1}: Shape mismatch")
