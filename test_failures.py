import numpy as np
import json
import sys
sys.path.insert(0, '.')
from ArcAgent_combined import ArcAgent
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcData import ArcData

def test_problem(problem_id, milestone):
    json_path = f'./Milestones/{milestone}/{problem_id}.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

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

    problem = ArcProblem(problem_id, trn_data, tst_data[0])

    agent = ArcAgent()
    predictions = agent.make_predictions(problem)

    expected = np.array(data['test'][0]['output'])

    print(f"\n{'='*80}")
    print(f"Testing {problem_id}")
    print(f"{'='*80}")
    print(f"Expected shape: {expected.shape}")
    print(f"Expected output:\n{expected}")

    for i, pred in enumerate(predictions, 1):
        print(f"\n--- Prediction {i} ---")
        print(f"Shape: {pred.shape}")
        print(f"Output:\n{pred}")
        if pred.shape == expected.shape:
            if np.array_equal(pred, expected):
                print(f"✓ EXACT MATCH!")
                return True
            else:
                matches = (pred == expected).sum()
                total = pred.size
                pct = 100.0 * matches / total
                print(f"  {pct:.1f}% match ({matches}/{total} pixels)")
                diff = np.where(pred != expected)
                print(f"  Differences at positions: {list(zip(diff[0][:5], diff[1][:5]))[:5]}")
        else:
            print(f"  Shape mismatch")

    return False

# Test the 3 failed Milestone B problems
print("Testing Milestone B failures:")
test_problem('28e73c20', 'B')
test_problem('0520fde7', 'B')
test_problem('6430c8c4', 'B')
