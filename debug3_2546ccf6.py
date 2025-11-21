import json
import os.path
import numpy as np

# Load the problem
problem_name = "2546ccf6.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    test_input = np.array(flat_data['test'][0]['input'])
    expected_output = np.array(flat_data['test'][0]['output'])

    # Find all color 1 cells
    print("Color 1 positions in INPUT:")
    color_1_coords = np.argwhere(test_input == 1)
    for r, c in color_1_coords:
        print(f"  ({r}, {c})")

    print("\nColor 1 positions in EXPECTED OUTPUT:")
    color_1_coords_out = np.argwhere(expected_output == 1)
    for r, c in color_1_coords_out:
        print(f"  ({r}, {c})")

    # Those are the cells we got wrong - they should be 0, not 1
    # So they're being incorrectly mirrored from somewhere
