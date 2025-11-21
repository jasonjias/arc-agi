import numpy as np
import json
from ArcAgent import rule_connect_blue_red_dots

# Load the test case
with open('./Milestones/D/992798f6.json', 'r') as f:
    data = json.load(f)

test_case = data['test'][0]
inp = np.array(test_case['input'])
expected = np.array(test_case['output'])

print("Input:")
print(inp)
print("\nExpected output:")
print(expected)

# Run the function
result = rule_connect_blue_red_dots(inp)

print("\nActual output:")
print(result)

print("\nDifferences:")
diff = (result != expected)
if np.any(diff):
    print("Positions where actual differs from expected:")
    diff_positions = np.argwhere(diff)
    for pos in diff_positions:
        r, c = pos
        print(f"  ({r}, {c}): expected {expected[r, c]}, got {result[r, c]}")
else:
    print("Perfect match!")

# Let's trace the path manually
blue_coords = np.where(inp == 1)
red_coords = np.where(inp == 2)
print(f"\nBlue at: ({blue_coords[0][0]}, {blue_coords[1][0]})")
print(f"Red at: ({red_coords[0][0]}, {red_coords[1][0]})")
