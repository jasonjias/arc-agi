import json
import os.path
import numpy as np

from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcAgent import find_enclosed_rings, rule_fill_enclosed_rings

# Load the problem
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Get test input
    test_input = np.array(flat_data['test'][0]['input'])
    expected_output = np.array(flat_data['test'][0]['output'])

print("Test Input Shape:", test_input.shape)
print("\nFinding enclosed rings...")

rings = find_enclosed_rings(test_input, bg=0)

print(f"\nFound {len(rings)} enclosed rings:")
for i, (ring_color, region_cells) in enumerate(rings):
    print(f"\nRing {i+1}: Color {ring_color}")
    print(f"  Region size: {len(region_cells)} cells")

    # Count colors inside
    color_counts = {}
    for r, c in region_cells:
        val = test_input[r, c]
        if val != 0 and val != ring_color:
            color_counts[val] = color_counts.get(val, 0) + 1

    print(f"  Interior colors: {color_counts}")
    if color_counts:
        fill_color = max(color_counts.keys(), key=lambda c: color_counts[c])
        print(f"  Would fill with: {fill_color}")
    else:
        print(f"  Would keep empty (no fill color)")

print("\n" + "="*60)
print("Running rule_fill_enclosed_rings...")
result = rule_fill_enclosed_rings(test_input)

print("\nResult shape:", result.shape)
print("Expected shape:", expected_output.shape)

if result.shape == expected_output.shape:
    matches = (result == expected_output).sum()
    total = expected_output.size
    print(f"Matches: {matches}/{total} ({matches/total*100:.1f}%)")

    diff = (result != expected_output)
    diff_coords = np.argwhere(diff)
    print(f"\nMismatched cells: {len(diff_coords)}")
    if len(diff_coords) > 0 and len(diff_coords) <= 30:
        print("\nDifferences (row, col) -> Expected vs Got:")
        for r, c in diff_coords:
            print(f"  ({r}, {c}): expected {expected_output[r, c]}, got {result[r, c]}")
