import numpy as np
import json

with open('./Milestones/C/f35d900a.json', 'r') as f:
    data = json.load(f)

# Use first training example
inp = np.array(data['train'][0]['input'])

# Find pixels
pixels = []
for r in range(inp.shape[0]):
    for c in range(inp.shape[1]):
        if inp[r, c] != 0:
            pixels.append((r, c, int(inp[r, c])))

print("Found pixels:", pixels)

# Group by color
color_groups = {}
for r, c, color in pixels:
    if color not in color_groups:
        color_groups[color] = []
    color_groups[color].append((r, c))

print("Color groups:", color_groups)

# Check connections
for color, positions in color_groups.items():
    (r1, c1), (r2, c2) = positions
    print(f"\nColor {color}: ({r1},{c1}) to ({r2},{c2})")

    if r1 == r2:
        print(f"  Horizontal connection at row {r1}")
        min_c = min(c1, c2)
        max_c = max(c1, c2)
        start_c = min_c + 2
        gap_cells = list(range(start_c, max_c - 1))
        print(f"  start_c={start_c}, max_c={max_c}, gap_cells={gap_cells}")

    if c1 == c2:
        print(f"  Vertical connection at col {c1}")
        min_r = min(r1, r2)
        max_r = max(r1, r2)
        start_r = min_r + 2
        gap_cells = list(range(start_r, max_r - 1))
        print(f"  start_r={start_r}, max_r={max_r}, gap_cells={gap_cells}")
