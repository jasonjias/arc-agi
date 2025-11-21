import numpy as np
import json

with open('./Milestones/C/f35d900a.json', 'r') as f:
    data = json.load(f)

# Training example 2
expected = np.array(data['train'][1]['output'])

# Check column 2 vertical line
print("Training example 2, col 2 values:")
for r in range(expected.shape[0]):
    val = expected[r, 2]
    if val != 0:
        print(f"  Row {r}: {val}")
