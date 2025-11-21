import json
import os.path
import numpy as np
from ArcAgent import find_enclosed_rings

# Load the problem
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    test_input = np.array(flat_data['test'][0]['input'])

print(f"Value at (15, 9): {test_input[15, 9]}")

rings = find_enclosed_rings(test_input, bg=0)

for i, (ring_color, region_cells) in enumerate(rings):
    if ring_color == 8:
        print(f"\nGray ring (color 8):")
        print(f"  Region size: {len(region_cells)} cells")
        print(f"  Is (15, 9) in region? {(15, 9) in region_cells}")

        region_list = list(region_cells)
        min_r = min(r for r, c in region_list)
        max_r = max(r for r, c in region_list)
        min_c = min(c for r, c in region_list)
        max_c = max(c for r, c in region_list)

        print(f"  Region bounding box: rows [{min_r}, {max_r}], cols [{min_c}, {max_c}]")
        print(f"  Is (15, 9) in bbox? {min_r <= 15 <= max_r and min_c <= 9 <= max_c}")
        print(f"  Is (15, 9) on bbox perimeter? {15 == min_r or 15 == max_r or 9 == min_c or 9 == max_c}")
        print(f"  Is (15, 9) strictly interior? {min_r < 15 < max_r and min_c < 9 < max_c}")
