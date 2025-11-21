import json
import os.path
import numpy as np
from ArcAgent import rule_fit_shapes_into_holes

# Load the problem
problem_name = "67c52801.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    
    train_input = np.array(flat_data['train'][3]['input'])
    expected_output = np.array(flat_data['train'][3]['output'])
    
    print("Training Example 4:")
    print("Input:")
    print(train_input)
    print("\nExpected Output:")
    print(expected_output)
    
    result = rule_fit_shapes_into_holes(train_input)
    
    print("\nMy Output:")
    print(result)
    
    matches = np.array_equal(result, expected_output)
    if result.shape == expected_output.shape:
        match_pct = np.sum(result == expected_output) / expected_output.size * 100
        print(f"\n{match_pct:.1f}% match - {'PASS' if matches else 'FAIL'}")
        
        if not matches:
            diff = (result != expected_output)
            print(f"\nDifferences at {np.sum(diff)} cells:")
            diff_coords = np.argwhere(diff)
            if len(diff_coords) <= 20:
                for r, c in diff_coords:
                    print(f"  ({r}, {c}): expected {expected_output[r, c]}, got {result[r, c]}")
    else:
        print(f"\nShape mismatch")
