import numpy as np
import json
from ArcAgent_milestoneB import ArcAgent

with open('./Milestones/B/28e73c20.json', 'r') as f:
    data = json.load(f)

# Test the spiral function directly
agent = ArcAgent()

# Test on a small example
test_input = np.zeros((6, 6), dtype=int)
result = agent._draw_green_spiral(test_input)

print("Test input (6x6):")
print(test_input)
print("\nGenerated spiral:")
print(result)
print("\nExpected spiral:")
expected = np.array(data['train'][0]['output'])
print(expected)
print("\nMatch?", np.array_equal(result, expected))
