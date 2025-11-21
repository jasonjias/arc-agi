import numpy as np
import json
from ArcAgent_milestoneC import rule_draw_boxes_and_connect_pixels

with open('./Milestones/C/f35d900a.json', 'r') as f:
    data = json.load(f)

# Use first training example
inp = np.array(data['train'][0]['input'])
expected = np.array(data['train'][0]['output'])

result = rule_draw_boxes_and_connect_pixels(inp)

print("Row 2 expected:", expected[2, :10])
print("Row 2 result  :", result[2, :10])
print()
print("Row 4 expected:", expected[4, :10])
print("Row 4 result  :", result[4, :10])
print()
print("Row 8 expected:", expected[8, :10])
print("Row 8 result  :", result[8, :10])
print()
print("Col 1 rows 2-9 expected:", expected[2:9, 1])
print("Col 1 rows 2-9 result  :", result[2:9, 1])
