import json
import os.path
import numpy as np

# Load the problem
problem_name = "2546ccf6.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Check first training example in detail
    train_input = np.array(flat_data['train'][0]['input'])
    train_output = np.array(flat_data['train'][0]['output'])

    print("Training Example 1 - Understanding the pattern:")
    h, w = train_input.shape

    # The grid divides into cells
    # H lines at [4, 9, 14] create rows: 0-3, 5-8, 10-13, 15-16
    # V lines at [4, 9, 14] create cols: 0-3, 5-8, 10-13, 15-18

    # Let's look at each cell
    print("\nQuadrant structure (each cell between grid lines):")
    print("Row 0-3, Col 0-3:")
    print(train_input[0:4, 0:4])

    print("\nRow 0-3, Col 5-8:")
    print(train_input[0:4, 5:9])

    print("\nRow 0-3, Col 10-13:")
    print(train_input[0:4, 10:14])

    print("\nRow 5-8, Col 0-3:")
    print(train_input[5:9, 0:4])

    print("\nRow 5-8, Col 5-8 (has pattern):")
    print(train_input[5:9, 5:9])

    print("\nRow 5-8, Col 10-13 (has pattern):")
    print(train_input[5:9, 10:14])

    print("\nRow 10-13, Col 5-8 (missing pattern - should match above):")
    print("INPUT:", train_input[10:14, 5:9])
    print("OUTPUT:", train_output[10:14, 5:9])

    print("\nRow 10-13, Col 10-13 (missing pattern):")
    print("INPUT:", train_input[10:14, 10:14])
    print("OUTPUT:", train_output[10:14, 10:14])

    # The pattern seems to be: complete the 4-way symmetry
    # Check if row 5-8 col 5-8 mirrors to row 10-13 col 5-8
    print("\n\nChecking vertical mirror:")
    print("Top cell (row 5-8, col 5-8):")
    print(train_output[5:9, 5:9])
    print("Bottom cell (row 10-13, col 5-8) should be vertically mirrored:")
    print(train_output[10:14, 5:9])

    # Check if they're vertical mirrors
    top = train_output[5:9, 5:9]
    bottom = train_output[10:14, 5:9]
    is_v_mirror = np.array_equal(top, bottom[::-1, :])
    print(f"Is vertical mirror: {is_v_mirror}")
