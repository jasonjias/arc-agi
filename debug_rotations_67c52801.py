import json
import os.path
import numpy as np

# Load the problem
problem_name = "67c52801.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    train_input = np.array(flat_data['train'][0]['input'])

    holes_row = 5
    w = train_input.shape[1]
    bg = 0

    print(f"Holes row: {train_input[holes_row, :]}")
    print(f"Holes at columns: {[c for c in range(w) if train_input[holes_row, c] == bg]}")

    # Shape 1 (color 2, 1x2)
    shape1 = np.array([[2, 2]])
    print(f"\nShape 1 original (1x2):")
    print(shape1)
    print(f"  Size: {shape1.shape}, non-zero: {np.count_nonzero(shape1)}")

    def get_rotations(arr):
        return [arr, np.rot90(arr), np.rot90(arr, 2), np.rot90(arr, 3)]

    print("\nShape 1 rotations:")
    for i, rot in enumerate(get_rotations(shape1)):
        print(f"  Rotation {i} (deg {i*90}): shape {rot.shape}")
        print(rot)

        # Try to find where this can be placed
        sh, sw = rot.shape
        print(f"    Looking for {sw} contiguous holes...")
        for start_col in range(w - sw + 1):
            all_holes = True
            for sc in range(sw):
                if train_input[holes_row, start_col + sc] != bg:
                    all_holes = False
                    break
            if all_holes:
                print(f"      Can place at column {start_col}!")
