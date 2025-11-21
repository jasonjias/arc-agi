import numpy as np
import json

# Load the test case
with open('./Milestones/D/e9b4f6fc.json', 'r') as f:
    data = json.load(f)

# First training case
case = data['train'][0]
inp = np.array(case['input'])
expected = np.array(case['output'])

print("Input shape:", inp.shape)
print("Expected output shape:", expected.shape)
print("\nInput:")
print(inp)
print("\nExpected output:")
print(expected)

# Find the frame (color 1)
frame_coords = np.argwhere(inp == 1)
if len(frame_coords) > 0:
    min_r, min_c = frame_coords.min(axis=0)
    max_r, max_c = frame_coords.max(axis=0)
    print(f"\nFrame bounds: rows {min_r}-{max_r}, cols {min_c}-{max_c}")
    print(f"Interior would be: rows {min_r+1}-{max_r-1}, cols {min_c+1}-{max_c-1}")

    interior = inp[min_r:max_r+1, min_c:max_c+1]
    print("\nFrame region:")
    print(interior)

# Look for legend pairs
print("\n\nLegend pairs (adjacent pixels outside frame):")
for r in range(inp.shape[0]):
    for c in range(inp.shape[1]):
        if inp[r, c] != 0:
            # Check if outside frame region
            if r < min_r or r > max_r or c < min_c or c > max_c:
                print(f"  ({r}, {c}): color {inp[r, c]}")
                # Check neighbors
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                        if inp[nr, nc] != 0 and inp[nr, nc] != inp[r, c]:
                            # Check if neighbor is also outside frame
                            if nr < min_r or nr > max_r or nc < min_c or nc > max_c:
                                print(f"    -> adjacent to ({nr}, {nc}): color {inp[nr, nc]}")
