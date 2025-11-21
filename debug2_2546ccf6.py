import json
import os.path
import numpy as np

# Load the problem
problem_name = "2546ccf6.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    test_input = np.array(flat_data['test'][0]['input'])

    # Look at the problematic region
    # Rows 0-3, columns around 14-20
    print("Test input region [0:4, 12:23]:")
    print(test_input[0:4, 12:23])

    print("\nTest input region [20:24, 12:23]:")
    print(test_input[20:24, 12:23])

    # What's the grid color?
    print("\nMost common color:")
    unique, counts = np.unique(test_input, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Color {val}: {count} cells")
