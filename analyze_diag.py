import numpy as np
import json

# Load the test case
with open('./Milestones/D/c1990cce.json', 'r') as f:
    data = json.load(f)

# Look at the third example
case = data['train'][2]
inp = np.array(case['input'])
expected = np.array(case['output'])

print("Input:")
print(inp)
red_col = np.where(inp[0] == 2)[0][0]
print(f"Red pixel at column: {red_col}")
print(f"Width: {inp.shape[1]}")

print("\n\nExpected output:")
for row_idx, row in enumerate(expected):
    print(f"Row {row_idx}: ", end="")
    for col_idx, val in enumerate(row):
        if val == 2:
            print(f"R", end="")
        elif val == 1:
            print(f"B", end="")
        else:
            print(f".", end="")
    red_cols = [i for i, v in enumerate(row) if v == 2]
    blue_cols = [i for i, v in enumerate(row) if v == 1]
    print(f"  Red:{red_cols} Blue:{blue_cols}")

print("\n\nBlue diagonal lines:")
blue_positions = np.argwhere(expected == 1)
# Group by diagonal (col - row should be constant for same diagonal)
diagonals = {}
for pos in blue_positions:
    diag_id = pos[1] - pos[0]  # column - row
    if diag_id not in diagonals:
        diagonals[diag_id] = []
    diagonals[diag_id].append((pos[0], pos[1]))

for diag_id, positions in sorted(diagonals.items()):
    print(f"\nDiagonal {diag_id} (col - row = {diag_id}):")
    for pos in positions:
        print(f"  ({pos[0]}, {pos[1]})")
