import numpy as np
import json

with open('./Milestones/C/f35d900a.json', 'r') as f:
    data = json.load(f)

# Test case
expected = np.array(data['test'][0]['output'])

print("Test case expected, col 2:")
for r in range(expected.shape[0]):
    val = expected[r, 2]
    if val == 5:
        print(f"  Row {r}: dot")
    elif val != 0:
        print(f"  Row {r}: box/pixel (color {val})")

print("\nTest case expected, col 12:")
for r in range(expected.shape[0]):
    val = expected[r, 12]
    if val == 5:
        print(f"  Row {r}: dot")
    elif val != 0:
        print(f"  Row {r}: box/pixel (color {val})")
