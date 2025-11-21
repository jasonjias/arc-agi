import numpy as np
import json
import sys
sys.path.insert(0, '.')
from ArcAgent import rule_color_swap_with_legend

# Load the test case
with open('./Milestones/D/e9b4f6fc.json', 'r') as f:
    data = json.load(f)

all_pass = True
for idx, case in enumerate(data['train']):
    inp = np.array(case['input'])
    expected = np.array(case['output'])
    result = rule_color_swap_with_legend(inp)

    match = np.array_equal(result, expected)
    print(f"Training case {idx}: {'PASS' if match else 'FAIL'}")

    if not match:
        all_pass = False
        print(f"  Expected shape: {expected.shape}, Got shape: {result.shape}")
        print(f"  Input shape: {inp.shape}")
        if result.shape == expected.shape:
            diff = (result != expected)
            if np.any(diff):
                print(f"  Differences found:")
                diff_positions = np.argwhere(diff)
                for pos in diff_positions[:10]:
                    r, c = pos
                    print(f"    ({r}, {c}): expected {expected[r, c]}, got {result[r, c]}")
        else:
            print(f"  Expected:")
            print(expected)
            print(f"  Got:")
            print(result)

# Test case
test_case = data['test'][0]
inp = np.array(test_case['input'])
expected = np.array(test_case['output'])
result = rule_color_swap_with_legend(inp)
match = np.array_equal(result, expected)
print(f"Test case: {'PASS' if match else 'FAIL'}")

if not match:
    all_pass = False
    print(f"  Expected shape: {expected.shape}, Got shape: {result.shape}")

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
