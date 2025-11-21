import json
import os.path
import numpy as np

# Load the problem
problem_name = "67c52801.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    train_input = np.array(flat_data['train'][0]['input'])
    expected_output = np.array(flat_data['train'][0]['output'])

    print("Input:")
    for i, row in enumerate(train_input):
        print(f"Row {i}: {row}")

    print("\nExpected Output:")
    for i, row in enumerate(expected_output):
        print(f"Row {i}: {row}")

    print("\n\nAnalysis:")
    print("Input row 5: [1 0 1 0 0 1] - This has holes at columns 1, 3, 4!")
    print("Input row 6: [1 1 1 1 1 1] - This is solid")
    print("\nOutput row 5: [1 2 1 3 3 1] - Shapes are embedded here!")
    print("Output row 4: [0 2 0 3 3 0] - Shapes extend above")
    print("\nSo the 'ground' is actually TWO rows: row 5 (with holes) and row 6 (solid base)")
    print("Shapes need to be placed starting from the holes in row 5, extending upward")
