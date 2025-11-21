import json
import os.path
import numpy as np

# Load the problem
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    test_input = np.array(flat_data['test'][0]['input'])

bg = 0

# Check each color for frame-like structure
for color in np.unique(test_input):
    if color == bg:
        continue

    color_coords = set(zip(*np.where(test_input == color)))
    print(f"\nColor {color}: {len(color_coords)} cells")

    if len(color_coords) < 8:
        print("  Too small to be a ring")
        continue

    rows = [r for r, c in color_coords]
    cols = [c for r, c in color_coords]

    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    # Check frame coverage
    top_count = sum((min_r, c) in color_coords for c in range(min_c, max_c + 1))
    bottom_count = sum((max_r, c) in color_coords for c in range(min_c, max_c + 1))
    left_count = sum((r, min_c) in color_coords for r in range(min_r, max_r + 1))
    right_count = sum((r, max_c) in color_coords for r in range(min_r, max_r + 1))

    width = max_c - min_c + 1
    height = max_r - min_r + 1

    print(f"  Bounding box: {width}x{height}")
    print(f"  Top coverage: {top_count}/{width} = {top_count/width*100:.1f}%")
    print(f"  Bottom coverage: {bottom_count}/{width} = {bottom_count/width*100:.1f}%")
    print(f"  Left coverage: {left_count}/{height} = {left_count/height*100:.1f}%")
    print(f"  Right coverage: {right_count}/{height} = {right_count/height*100:.1f}%")

    has_top = top_count >= width * 0.4
    has_bottom = bottom_count >= width * 0.4
    has_left = left_count >= height * 0.4
    has_right = right_count >= height * 0.4

    sides_present = sum([has_top, has_bottom, has_left, has_right])
    print(f"  Sides with >40% coverage: {sides_present}")

    if sides_present >= 3:
        print(f"  -> Would be considered a RING")
    else:
        print(f"  -> Would NOT be considered a ring")
