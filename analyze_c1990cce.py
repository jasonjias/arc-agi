import numpy as np
import json

# Load the test case
with open('./Milestones/D/c1990cce.json', 'r') as f:
    data = json.load(f)

# Look at the first training example in detail
case = data['train'][0]
inp = np.array(case['input'])
expected = np.array(case['output'])

print("Input:")
print(inp)
print(f"\nRed pixel at column: {np.where(inp[0] == 2)[0][0]}")
print(f"\nExpected output shape: {expected.shape}")
print("\nExpected output:")
for row_idx, row in enumerate(expected):
    print(f"Row {row_idx:2d}: ", end="")
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

# Analyze the blue diagonal pattern
print("\n\nBlue diagonal lines analysis:")
blue_positions = np.argwhere(expected == 1)
for pos in blue_positions[:20]:
    print(f"  Row {pos[0]}, Col {pos[1]}")
