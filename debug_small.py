import numpy as np
import json
import sys
sys.path.insert(0, '.')
from ArcAgent import rule_expanding_triangle_with_diagonals

# Load the test case
with open('./Milestones/D/c1990cce.json', 'r') as f:
    data = json.load(f)

# Test case 1 (small)
case = data['train'][1]
inp = np.array(case['input'])
expected = np.array(case['output'])
result = rule_expanding_triangle_with_diagonals(inp)

print("Expected:")
for row_idx, row in enumerate(expected):
    print(f"Row {row_idx}: ", end="")
    for val in row:
        if val == 2:
            print("R", end="")
        elif val == 1:
            print("B", end="")
        else:
            print(".", end="")
    print()

print("\nGot:")
for row_idx, row in enumerate(result):
    print(f"Row {row_idx}: ", end="")
    for val in row:
        if val == 2:
            print("R", end="")
        elif val == 1:
            print("B", end="")
        else:
            print(".", end="")
    print()

red_col = np.where(inp[0] == 2)[0][0]
max_dist = max(red_col, inp.shape[1] - 1 - red_col)
first_diag_id = (red_col - 1) - max_dist
print(f"\nred_col={red_col}, max_dist={max_dist}, first_diag_id={first_diag_id}")

# Expected blue at (3, 1) means col-row = 1-3 = -2
# Expected blue at (4, 2) means col-row = 2-4 = -2
print("\nExpected diagonal has col-row = -2")
print(f"My first_diag_id = {first_diag_id}")
print("They should match!")
