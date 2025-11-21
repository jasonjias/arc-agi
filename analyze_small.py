import numpy as np
import json

# Load the test case
with open('./Milestones/D/c1990cce.json', 'r') as f:
    data = json.load(f)

# Look at the second example (smaller one)
case = data['train'][1]
inp = np.array(case['input'])
expected = np.array(case['output'])

print("Input:")
print(inp)
print(f"Red pixel at column: {np.where(inp[0] == 2)[0][0]}")
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
    # Also show which columns have non-zero values
    red_cols = [i for i, v in enumerate(row) if v == 2]
    blue_cols = [i for i, v in enumerate(row) if v == 1]
    print(f"  Red:{red_cols} Blue:{blue_cols}")

print("\n\nDiagonal analysis:")
blue_positions = np.argwhere(expected == 1)
print("Blue positions (row, col):")
for pos in blue_positions:
    print(f"  ({pos[0]}, {pos[1]})")

# Look for diagonal pattern - difference between consecutive positions
if len(blue_positions) > 1:
    print("\nDiagonal direction (delta row, delta col):")
    for i in range(len(blue_positions) - 1):
        dr = blue_positions[i+1][0] - blue_positions[i][0]
        dc = blue_positions[i+1][1] - blue_positions[i][1]
        print(f"  From ({blue_positions[i][0]},{blue_positions[i][1]}) to ({blue_positions[i+1][0]},{blue_positions[i+1][1]}): ({dr}, {dc})")
