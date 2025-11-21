import numpy as np
import json

# Load the test case
with open('./Milestones/D/c1990cce.json', 'r') as f:
    data = json.load(f)

# Look at first example
case = data['train'][0]
inp = np.array(case['input'])
expected = np.array(case['output'])

print("Input (1D):", inp[0])
print("Width:", len(inp[0]))

# Find the red pixel
red_pos = np.where(inp[0] == 2)[0][0]
print(f"Red pixel at index: {red_pos}")
print(f"Distance to left edge: {red_pos}")
print(f"Distance to right edge: {len(inp[0]) - 1 - red_pos}")

print("\nExpected output:")
for i, row in enumerate(expected):
    print(f"Row {i:2d}: ", end="")
    for j, val in enumerate(row):
        if val == 2:
            print('R', end='')
        elif val == 1:
            print('B', end='')
        else:
            print('.', end='')
    print()

# Analyze red triangle
print("\n\nRed pixels positions:")
for i, row in enumerate(expected):
    red_in_row = np.where(row == 2)[0]
    if len(red_in_row) > 0:
        print(f"Row {i}: positions {list(red_in_row)}")

# Analyze blue diagonals
print("\n\nBlue pixels positions:")
for i, row in enumerate(expected):
    blue_in_row = np.where(row == 1)[0]
    if len(blue_in_row) > 0:
        print(f"Row {i}: positions {list(blue_in_row)}")

# Look for the pattern
print("\n\n=== Second example ===")
case2 = data['train'][1]
inp2 = np.array(case2['input'])
expected2 = np.array(case2['output'])

print("Input (1D):", inp2[0])
print("Width:", len(inp2[0]))
red_pos2 = np.where(inp2[0] == 2)[0][0]
print(f"Red pixel at index: {red_pos2}")

print("\nExpected output:")
for i, row in enumerate(expected2):
    print(f"Row {i:2d}: ", end="")
    for j, val in enumerate(row):
        if val == 2:
            print('R', end='')
        elif val == 1:
            print('B', end='')
        else:
            print('.', end='')
    print()

print("\n\nRed pixels positions:")
for i, row in enumerate(expected2):
    red_in_row = np.where(row == 2)[0]
    if len(red_in_row) > 0:
        print(f"Row {i}: positions {list(red_in_row)}")

print("\n\nBlue pixels positions:")
for i, row in enumerate(expected2):
    blue_in_row = np.where(row == 1)[0]
    if len(blue_in_row) > 0:
        print(f"Row {i}: positions {list(blue_in_row)}")
