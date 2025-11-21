import json
import os.path
import numpy as np
from ArcAgent import rule_draw_rings_around_blue_ring

# Load the problem
problem_name = "d931c21c.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    test_input = np.array(flat_data['test'][0]['input'])
    expected_output = np.array(flat_data['test'][0]['output'])

print("Test Input shape:", test_input.shape)
print("Expected Output shape:", expected_output.shape)

result = rule_draw_rings_around_blue_ring(test_input)

print("\nMy Output shape:", result.shape)

if result.shape == expected_output.shape:
    diff = (result != expected_output)
    print(f"\nMatches: {np.sum(result == expected_output)}/{expected_output.size}")
    print(f"Mismatches: {np.sum(diff)}")

    diff_coords = np.argwhere(diff)
    if len(diff_coords) > 0 and len(diff_coords) <= 50:
        print("\nDifferences (row, col) -> Expected vs Got:")
        for r, c in diff_coords:
            print(f"  ({r}, {c}): expected {expected_output[r, c]}, got {result[r, c]}")
