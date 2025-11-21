import numpy as np
import json
import sys
sys.path.insert(0, '.')
from ArcAgent import ArcAgent
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcData import ArcData

# Load the problem
with open('./Milestones/D/c1990cce.json', 'r') as f:
    data = json.load(f)

# Convert to ArcProblem format
trn_data = []
for dt in data['train']:
    d_input = ArcData(np.array(dt['input']))
    d_output = ArcData(np.array(dt['output']))
    trn_set = ArcSet(arc_input=d_input, arc_output=d_output)
    trn_data.append(trn_set)

tst_data = []
for tst in data['test']:
    t_input = ArcData(np.array(tst['input']))
    t_output = ArcData(np.array(tst['output']))
    tst_set = ArcSet(arc_input=t_input, arc_output=t_output)
    tst_data.append(tst_set)

# Use first test set
problem = ArcProblem("c1990cce", trn_data, tst_data[0])

# Create agent and make predictions
agent = ArcAgent()
predictions = agent.make_predictions(problem)

# Get the expected output
expected = np.array(data['test'][0]['output'])

# Check predictions
print("Testing c1990cce.json")
print("=" * 80)
for i, pred in enumerate(predictions, 1):
    if pred.shape == expected.shape:
        if np.array_equal(pred, expected):
            print(f"✓ Prediction {i}: EXACT MATCH!")
            print("SUCCESS!")
            sys.exit(0)
        else:
            matches = (pred == expected).sum()
            total = pred.size
            pct = 100.0 * matches / total
            print(f"  Prediction {i}: {pct:.1f}% match ({matches}/{total} pixels)")
    else:
        print(f"  Prediction {i}: Shape mismatch ({pred.shape} vs {expected.shape})")

print("\nFAILED - No exact match found")
sys.exit(1)
