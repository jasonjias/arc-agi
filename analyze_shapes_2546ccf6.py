import json
import os.path
import numpy as np

# Load the problem
problem_name = "2546ccf6.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Check first training example
    train_input = np.array(flat_data['train'][0]['input'])
    train_output = np.array(flat_data['train'][0]['output'])

    print("Training Example 1:")
    print("Looking for the pattern in each 'cell' between grid lines")

    # Grid lines at rows [4, 9, 14] and cols [4, 9, 14]
    # So we have a 4x4 grid of cells

    # Look at the center 2x2 arrangement (rows 5-13, cols 5-13)
    # This is split by line at 9, so we have:
    # Top-left: rows 5-8, cols 5-8
    # Top-right: rows 5-8, cols 10-13
    # Bottom-left: rows 10-13, cols 5-8
    # Bottom-right: rows 10-13, cols 10-13

    print("\nTop-left (5-8, 5-8):")
    print(train_input[5:9, 5:9])

    print("\nTop-right (5-8, 10-13):")
    print(train_input[5:9, 10:14])

    print("\nBottom-left (10-13, 5-8):")
    print(train_input[10:14, 5:9])

    print("\nBottom-right (10-13, 10-13) - MISSING:")
    print(train_input[10:14, 10:14])

    print("\nBottom-right OUTPUT (10-13, 10-13):")
    print(train_output[10:14, 10:14])

    # Check if bottom-right should be:
    # - horizontal flip of bottom-left?
    # - vertical flip of top-right?
    # - both flips of top-left?

    top_left = train_output[5:9, 5:9]
    top_right = train_output[5:9, 10:14]
    bottom_left = train_output[10:14, 5:9]
    bottom_right = train_output[10:14, 10:14]

    print("\nChecking relationships:")
    print(f"Top-right == fliplr(Top-left): {np.array_equal(top_right, np.fliplr(top_left))}")
    print(f"Bottom-left == flipud(Top-left): {np.array_equal(bottom_left, np.flipud(top_left))}")
    print(f"Bottom-right == flipud(Top-right): {np.array_equal(bottom_right, np.flipud(top_right))}")
    print(f"Bottom-right == fliplr(Bottom-left): {np.array_equal(bottom_right, np.fliplr(bottom_left))}")
    print(f"Bottom-right == flipud(fliplr(Top-left)): {np.array_equal(bottom_right, np.flipud(np.fliplr(top_left)))}")
