import numpy as np
import json
import os.path

# Load the problem
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    test_input = np.array(flat_data['test'][0]['input'])

# Check what's around (15, 9)
r, c = 15, 9
print(f"Position (15, 9): value = {test_input[r, c]}")
print("\nNeighbors:")
for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1,1), (1,-1), (-1,1), (-1,-1)]:
    nr, nc = r + dr, c + dc
    if 0 <= nr < test_input.shape[0] and 0 <= nc < test_input.shape[1]:
        print(f"  ({nr}, {nc}): {test_input[nr, nc]}")

# Show the region around (15, 9)
print(f"\nRegion around (15, 9):")
for rr in range(13, 18):
    row_str = ""
    for cc in range(7, 12):
        val = test_input[rr, cc]
        if (rr, cc) == (15, 9):
            row_str += f"[{val}] "
        else:
            row_str += f" {val}  "
    print(f"Row {rr}: {row_str}")
