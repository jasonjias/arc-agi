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
    print(f"Testing {problem_id} (Milestone {milestone})")
    print(f"{'='*80}")

    matched = False
    for i, pred in enumerate(predictions, 1):
        if pred.shape == expected.shape and np.array_equal(pred, expected):
            print(f"✓ EXACT MATCH on prediction {i}!")
            matched = True
            break

    if not matched:
        print(f"✗ NO MATCH")
        print(f"Expected shape: {expected.shape}")
        for i, pred in enumerate(predictions[:1], 1):
            print(f"\nPrediction {i} shape: {pred.shape}")
            if pred.shape == expected.shape:
                matches = (pred == expected).sum()
                total = pred.size
                pct = 100.0 * matches / total
                print(f"  {pct:.1f}% match ({matches}/{total} pixels)")

    return matched

# Test Milestone C failures
print("="*80)
print("MILESTONE C FAILURES")
print("="*80)
test_problem('b2862040', 'C')
test_problem('3428a4f5', 'C')
test_problem('7b6016b9', 'C')

# Test some Milestone D failures
print("\n" + "="*80)
print("MILESTONE D FAILURES")
print("="*80)
test_problem('18419cfa', 'D')
test_problem('bcb3040b', 'D')
test_problem('e9b4f6fc', 'D')
test_problem('d931c21c', 'D')
test_problem('c1990cce', 'D')
