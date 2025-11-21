import numpy as np
import json
from ArcAgent_milestoneC import rule_draw_boxes_and_connect_pixels

with open('./Milestones/C/f35d900a.json', 'r') as f:
    data = json.load(f)

# Use first training example
inp = np.array(data['train'][0]['input'])
expected = np.array(data['train'][0]['output'])

result = rule_draw_boxes_and_connect_pixels(inp)

print("Input shape:", inp.shape)
print("Expected shape:", expected.shape)
print("Result shape:", result.shape)
print()

# Compare
matches = (result == expected).sum()
total = result.size
print(f"Match: {matches}/{total} = {100*matches/total:.1f}%")
print()

# Find differences
diffs = []
for r in range(result.shape[0]):
    for c in range(result.shape[1]):
        if result[r, c] != expected[r, c]:
            diffs.append((r, c, expected[r, c], result[r, c]))

print(f"Found {len(diffs)} differences:")
for r, c, exp, got in diffs[:20]:  # Show first 20
    print(f"  ({r},{c}): expected {exp}, got {got}")
