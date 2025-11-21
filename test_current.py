import numpy as np
import json
import sys
sys.path.insert(0, '.')
from ArcAgent import rule_connect_blue_red_dots

# Load the test case
with open('./Milestones/D/992798f6.json', 'r') as f:
    data = json.load(f)

test_case = data['test'][0]
inp = np.array(test_case['input'])
expected = np.array(test_case['output'])

result = rule_connect_blue_red_dots(inp)

# Find green in result
result_green = np.where(result == 3)
print(f"Actual green path ({len(result_green[0])} pixels):")
for i in range(len(result_green[0])):
    r, c = result_green[0][i], result_green[1][i]
    print(f"  ({r}, {c})")

print("\nExpected green path:")
expected_green = np.where(expected == 3)
for i in range(len(expected_green[0])):
    r, c = expected_green[0][i], expected_green[1][i]
    print(f"  ({r}, {c})")

print("\nMatch:", np.array_equal(result, expected))
