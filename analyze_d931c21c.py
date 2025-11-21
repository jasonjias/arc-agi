import json
import os.path
import numpy as np

# Load the problem
problem_name = "d931c21c.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Look at first training example
    train_input = np.array(flat_data['train'][0]['input'])
    train_output = np.array(flat_data['train'][0]['output'])

    print("Training Example 1:")
    print("Input:")
    print(train_input)
    print("\nOutput:")
    print(train_output)

    # Find where orange (2) appears in output but not input
    orange_mask = (train_output == 2) & (train_input != 2)
    print("\nOrange ring positions (new in output):")
    orange_coords = np.argwhere(orange_mask)
    for r, c in orange_coords:
        print(f"  ({r}, {c})")
        # Check what's adjacent
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < train_input.shape[0] and 0 <= nc < train_input.shape[1]:
                    neighbors.append((train_input[nr, nc], dr, dc))
        blue_neighbors = [(v, dr, dc) for v, dr, dc in neighbors if v == 1]
        print(f"    Blue neighbors: {blue_neighbors}")
