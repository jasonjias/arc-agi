import json
import os.path
import numpy as np

# Load the problem
problem_name = "2546ccf6.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    train_input = np.array(flat_data['train'][1]['input'])
    train_output = np.array(flat_data['train'][1]['output'])

    print("Training Example 2:")
    print("What changed from input to output:")

    diff = (train_input != train_output)
    diff_coords = np.argwhere(diff)

    print(f"\nTotal changes: {len(diff_coords)}")

    # Group by color changed TO
    changes_by_color = {}
    for r, c in diff_coords:
        color_to = train_output[r, c]
        if color_to not in changes_by_color:
            changes_by_color[color_to] = []
        changes_by_color[color_to].append((r, c))

    for color, coords in changes_by_color.items():
        print(f"\nColor {color}: {len(coords)} cells added")
        for r, c in coords:
            print(f"  ({r}, {c}): {train_input[r, c]} -> {train_output[r, c]}")
