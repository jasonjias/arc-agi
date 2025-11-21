import numpy as np
import json

# Load the test case
with open('./Milestones/D/992798f6.json', 'r') as f:
    data = json.load(f)

test_case = data['test'][0]
inp = np.array(test_case['input'])
expected = np.array(test_case['output'])

# Find blue and red
blue_coords = np.where(inp == 1)
red_coords = np.where(inp == 2)
blue_r, blue_c = int(blue_coords[0][0]), int(blue_coords[1][0])
red_r, red_c = int(red_coords[0][0]), int(red_coords[1][0])

print(f"Blue at: ({blue_r}, {blue_c})")
print(f"Red at: ({red_r}, {red_c})")
print(f"Distance: row={red_r - blue_r}, col={red_c - blue_c}")

# Extract the green path from expected output
green_coords = np.where(expected == 3)
print(f"\nExpected green path ({len(green_coords[0])} pixels):")
for i in range(len(green_coords[0])):
    r, c = green_coords[0][i], green_coords[1][i]
    print(f"  ({r}, {c})")

# Trace what the pattern is
print("\nPattern analysis:")
print("Looking at the path, it appears to:")
print("1. Move diagonally from blue")
print("2. When one dimension is within 1 unit of red, go straight")
