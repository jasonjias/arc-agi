import json
import os.path
import numpy as np

from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcAgent import ArcAgent

# Load the problem
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # convert the data into ArcData
    trn_data = []
    for dt in flat_data['train']:
        d_input = ArcData(np.array(dt['input']))
        d_output = ArcData(np.array(dt['output']))
        trn_set = ArcSet(arc_input=d_input, arc_output=d_output)
        trn_data.append(trn_set)

    tst_data = []
    for tst in flat_data['test']:
        t_input = ArcData(np.array(tst['input']))
        t_output = ArcData(np.array(tst['output']))
        tst_set = ArcSet(arc_input=t_input, arc_output=t_output)
        tst_data.append(tst_set)

    arc_problem = ArcProblem(problem_name[:-5], trn_data, tst_data[0])

# Run the agent
agent = ArcAgent()
print(f"Testing problem: {problem_name}")
print("="*60)

# Get test input
test_input = arc_problem.test_set().get_input_data().data()
print("\nTest Input:")
print(test_input)

# Get expected output
answer = arc_problem.test_set().get_output_data().data()
print("\nExpected Output:")
print(answer)

# Get predictions
preds = agent.make_predictions(arc_problem)

# Show first prediction with differences
print("\nPrediction 1:")
print(preds[0])

# Show differences
if preds[0].shape == answer.shape:
    diff = (preds[0] != answer)
    print("\nDifferences (True = mismatch):")
    print(diff)
    print(f"\nMismatched cells: {np.sum(diff)} out of {answer.size}")

    # Show which cells differ
    diff_coords = np.argwhere(diff)
    if len(diff_coords) > 0:
        print("\nMismatched positions (row, col) -> Expected vs Got:")
        for r, c in diff_coords:
            print(f"  ({r}, {c}): expected {answer[r, c]}, got {preds[0][r, c]}")
