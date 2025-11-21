import numpy as np
import json

with open('./Milestones/C/f35d900a.json', 'r') as f:
    data = json.load(f)

# Use test case
inp = np.array(data['test'][0]['input'])

# Find pixels
pixels = []
for r in range(inp.shape[0]):
    for c in range(inp.shape[1]):
        if inp[r, c] != 0:
            pixels.append((r, c, int(inp[r, c])))

print("Found pixels:", pixels)

# For vertical lines at cols 2 and 12
col2_pixels = [(r, c, clr) for r, c, clr in pixels if c == 2]
col12_pixels = [(r, c, clr) for r, c, clr in pixels if c == 12]

print("\nCol 2 pixels:", col2_pixels)
print("Col 12 pixels:", col12_pixels)

if len(col2_pixels) == 2:
    r1, _, _ = col2_pixels[0]
    r2, _, _ = col2_pixels[1]
    print(f"\nFor col 2: connecting rows {r1} and {r2}")
    print(f"  start_r = {r1} + 2 = {r1+2}")
    print(f"  end_r = {r2} - 1 = {r2-1}")
    print(f"  range({r1+2}, {r2-1}) = {list(range(r1+2, r2-1))}")
