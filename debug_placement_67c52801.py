import json
import os.path
import numpy as np

# Load the problem
problem_name = "67c52801.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    train_input = np.array(flat_data['train'][0]['input'])

    h, w = train_input.shape
    bg = 0

    # Find base and holes
    base_row = 6
    ground_color = 1
    holes_row = 5

    print(f"Base row: {base_row}")
    print(f"Holes row: {holes_row}")
    print(f"Holes row content: {train_input[holes_row, :]}")
    print(f"Holes at columns: {[c for c in range(w) if train_input[holes_row, c] == bg]}")

    # Shapes
    print("\nShape 1 (color 2): 1x2")
    shape1 = np.array([[2, 2]])
    print(shape1)
    print(f"  Need 2 contiguous holes for shape width=2")
    print(f"  Column 1: hole? {train_input[holes_row, 1] == bg}")
    print(f"  Column 2: hole? {train_input[holes_row, 2] == bg}")
    print(f"  Can place at col 1? NO (col 2 is ground color 1, not bg)")
    print(f"  Column 3: hole? {train_input[holes_row, 3] == bg}")
    print(f"  Column 4: hole? {train_input[holes_row, 4] == bg}")
    print(f"  Can place at col 3? YES (cols 3-4 are both holes)")

    print("\nShape 2 (color 3): 2x2")
    shape2 = np.array([[3, 3], [3, 3]])
    print(shape2)
    print(f"  Need 2 contiguous holes for shape width=2")
    print(f"  Can place at col 3? YES (but already used by shape 1 if placed there)")

    print("\nThe issue is that both shapes want columns 3-4!")
    print("We need a better placement algorithm that tries all combinations")
    print("\nExpected placement:")
    print("  Shape 1 (color 2, 1x2) at col 1")
    print("    But col 1-2 is NOT both holes (col 2 = 1)")
    print("    Wait... let me check if shape is actually 2x1 (vertical)!")

    shape1_rotated = np.rot90(shape1)
    print(f"\nShape 1 rotated 90°:")
    print(shape1_rotated)
    print(f"  Size: {shape1_rotated.shape} (2 rows, 1 col)")
    print(f"  Can place at col 1? YES (col 1 is a hole, shape width=1)")
