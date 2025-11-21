import json
import os.path
import numpy as np
from ArcAgent import rule_complete_quadrant_symmetry

# Load the problem
problem_name = "2546ccf6.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Check first training example
    train_input = np.array(flat_data['train'][0]['input'])
    train_output = np.array(flat_data['train'][0]['output'])

    print("Training Example 1:")
    print("Input shape:", train_input.shape)
    print("Output shape:", train_output.shape)

    # Find grid lines
    h, w = train_input.shape
    grid_color = 6

    h_lines = []
    for r in range(h):
        if all(train_input[r, c] == grid_color for c in range(w)):
            h_lines.append(r)

    v_lines = []
    for c in range(w):
        if all(train_input[r, c] == grid_color for r in range(h)):
            v_lines.append(c)

    print(f"\nHorizontal grid lines: {h_lines}")
    print(f"Vertical grid lines: {v_lines}")

    # Check what changed from input to output
    diff = (train_input != train_output)
    diff_coords = np.argwhere(diff)

    print(f"\nCells that changed: {len(diff_coords)}")
    if len(diff_coords) <= 30:
        print("Changes (row, col) -> Input to Output:")
        for r, c in diff_coords:
            print(f"  ({r}, {c}): {train_input[r, c]} -> {train_output[r, c]}")
