import numpy as np
import json
import sys
sys.path.insert(0, '.')
from ArcAgent import rule_expanding_triangle_with_diagonals

with open('./Milestones/D/c1990cce.json', 'r') as f:
    data = json.load(f)

case = data['train'][0]
inp = np.array(case['input'])
expected = np.array(case['output'])
result = rule_expanding_triangle_with_diagonals(inp)

print("Visual comparison:")
for i in range(len(expected)):
    exp_row = "".join("R" if v==2 else "B" if v==1 else "." for v in expected[i])
    got_row = "".join("R" if v==2 else "B" if v==1 else "." for v in result[i])
    marker = " <-- DIFF" if exp_row.__ne__(got_row) else ""
    print(f"{i:2d}: E:{exp_row} G:{got_row}{marker}")
