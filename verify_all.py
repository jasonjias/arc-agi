import numpy as np
import json
import sys
sys.path.insert(0, '.')
from ArcAgent import rule_connect_blue_red_dots

# Load all cases
with open('./Milestones/D/992798f6.json', 'r') as f:
    data = json.load(f)

all_pass = True
for idx, case in enumerate(data['train']):
    inp = np.array(case['input'])
    expected = np.array(case['output'])
    result = rule_connect_blue_red_dots(inp)

    match = np.array_equal(result, expected)
    print(f"Training case {idx}: {'PASS' if match else 'FAIL'}")

    if not match:
        all_pass = False
        blue_coords = np.where(inp == 1)
        red_coords = np.where(inp == 2)
        print(f"  Blue: ({blue_coords[0][0]}, {blue_coords[1][0]})")
        print(f"  Red: ({red_coords[0][0]}, {red_coords[1][0]})")

# Test case
test_case = data['test'][0]
inp = np.array(test_case['input'])
expected = np.array(test_case['output'])
result = rule_connect_blue_red_dots(inp)
match = np.array_equal(result, expected)
print(f"Test case: {'PASS' if match else 'FAIL'}")

if not match:
    all_pass = False

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
