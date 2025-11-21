import json
import os.path
import numpy as np

from ArcAgent import rule_fill_frames_with_frequent_color

# Load the problem
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Check first training example
    dt = flat_data['train'][0]
    inp = np.array(dt['input'])
    expected_out = np.array(dt['output'])

    predicted = rule_fill_frames_with_frequent_color(inp)

    print("=== Training Example 1 - Differences ===")
    diff_locations = np.where(predicted != expected_out)

    print(f"Total differences: {len(diff_locations[0])}")

    for i in range(min(10, len(diff_locations[0]))):
        r, c = diff_locations[0][i], diff_locations[1][i]
        print(f"  ({r}, {c}): predicted={predicted[r, c]}, expected={expected_out[r, c]}, input={inp[r, c]}")
