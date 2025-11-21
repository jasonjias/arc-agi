import numpy as np
import json

# Load the test case
with open('./Milestones/D/c48954c1.json', 'r') as f:
    data = json.load(f)

# First training case
case = data['train'][0]
inp = np.array(case['input'])
expected = np.array(case['output'])

print("Input:")
print(inp)

print("\nRotated 180:")
rotated = np.rot90(inp, 2)
print(rotated)

print("\nExpected output (as 3x3 tiles):")
for tile_row in range(3):
    print(f"\nRow {tile_row}:")
    for r in range(3):
        for tile_col in range(3):
            for c in range(3):
                val = expected[tile_row * 3 + r, tile_col * 3 + c]
                print(val, end=" ")
            print("| ", end="")
        print()

print("\n\nLet me check what pattern each tile follows:")
print("Tile [0,0] (top-left):")
print(expected[0:3, 0:3])
print("\nTile [0,1] (top-middle):")
print(expected[0:3, 3:6])
print("\nTile [0,2] (top-right):")
print(expected[0:3, 6:9])
print("\nTile [1,0] (middle-left):")
print(expected[3:6, 0:3])
print("\nTile [1,1] (middle-middle):")
print(expected[3:6, 3:6])

print("\n\nRotated:")
print(rotated)
print("\nFlip UD of rotated:")
print(np.flipud(rotated))
print("\nFlip LR of rotated:")
print(np.fliplr(rotated))
