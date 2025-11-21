import numpy as np
import json

# Load the test case
with open('./Milestones/D/c1990cce.json', 'r') as f:
    data = json.load(f)

# Look at the first example
case = data['train'][0]
expected = np.array(case['output'])

print("Blue diagonal lines:")
blue_positions = np.argwhere(expected == 1)
# Group by diagonal (col - row should be constant for same diagonal)
diagonals = {}
for pos in blue_positions:
    diag_id = pos[1] - pos[0]  # column - row
    if diag_id not in diagonals:
        diagonals[diag_id] = []
    diagonals[diag_id].append((pos[0], pos[1]))

for diag_id, positions in sorted(diagonals.items()):
    print(f"\nDiagonal {diag_id} (col - row = {diag_id}):")
    for pos in positions:
        print(f"  ({pos[0]}, {pos[1]})")
