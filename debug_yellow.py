import json
import os.path
import numpy as np

# Load the problem
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    test_input = np.array(flat_data['test'][0]['input'])

# Check yellow (color 7) ring
print("Yellow (color 7) positions:")
yellow_coords = list(zip(*np.where(test_input == 7)))
print(f"Found {len(yellow_coords)} yellow cells")
for r, c in sorted(yellow_coords):
    print(f"  ({r}, {c})")

# Check the bounding box
if yellow_coords:
    rows = [r for r, c in yellow_coords]
    cols = [c for r, c in yellow_coords]
    print(f"\nBounding box: rows [{min(rows)}, {max(rows)}], cols [{min(cols)}, {max(cols)}]")

    # Check if it forms a frame
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    print(f"\nChecking if forms a complete frame...")
    # Check if corners exist
    corners = [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]
    for corner in corners:
        if corner in yellow_coords:
            print(f"  Corner {corner}: YES")
        else:
            print(f"  Corner {corner}: NO - MISSING!")

    # Show what's inside the potential ring
    print(f"\nInside the yellow ring (rows {min_r+1}-{max_r-1}, cols {min_c+1}-{max_c-1}):")
    for r in range(min_r+1, max_r):
        for c in range(min_c+1, max_c):
            val = test_input[r, c]
            if val != 0:
                print(f"  ({r}, {c}): {val}")
