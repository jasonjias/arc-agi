import json
import os.path
import numpy as np

from ArcAgent import find_enclosed_regions_by_color

# Load the problem
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Look at first training example
    dt = flat_data['train'][0]
    inp = np.array(dt['input'])
    expected_out = np.array(dt['output'])

    print("=== Finding Enclosed Regions ===")
    enclosed = find_enclosed_regions_by_color(inp, bg=0)

    print(f"Found {len(enclosed)} enclosed regions\n")

    for frame_color, region_cells in enclosed:
        print(f"Frame color {frame_color}: {len(region_cells)} cells enclosed")

        # Count what's inside
        color_counts = {}
        for r, c in region_cells:
            val = inp[r, c]
            if val != 0 and val != frame_color:
                color_counts[val] = color_counts.get(val, 0) + 1

        print(f"  Colors inside: {color_counts}")

        # Show bounding box
        rs = [r for r, c in region_cells]
        cs = [c for r, c in region_cells]
        if rs and cs:
            print(f"  Bounding box: rows {min(rs)}-{max(rs)}, cols {min(cs)}-{max(cs)}")
        print()
