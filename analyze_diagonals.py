import numpy as np
import json

with open('./Milestones/D/c1990cce.json', 'r') as f:
    data = json.load(f)

case = data['train'][0]
expected = np.array(case['output'])

# Group blue pixels by their diagonal (col - row)
blue_pixels = []
for r in range(len(expected)):
    for c in range(len(expected[r])):
        if expected[r][c] == 1:
            blue_pixels.append((r, c, c-r))

# Group by diagonal ID
from collections import defaultdict
diagonals = defaultdict(list)
for r, c, diag_id in blue_pixels:
    diagonals[diag_id].append((r, c))

print("Blue diagonals grouped by (col - row):")
for diag_id in sorted(diagonals.keys()):
    positions = diagonals[diag_id]
    print(f"  Diagonal {diag_id:3d}: {positions}")
    if len(positions) > 0:
        start_r, start_c = positions[0]
        print(f"      Starts at ({start_r}, {start_c})")

red_col = 6
max_dist = 6
print(f"\nRed center: col {red_col}, max_dist: {max_dist}")
print(f"Middle row (where red reaches edges): {max_dist}")
