import json
import os.path
import numpy as np

from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcAgent import ArcAgent

# Load the problem
problem_name = "d931c21c.json"
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
preds = agent.make_predictions(arc_problem)

# Check if correct
answer = arc_problem.test_set().get_output_data().data()
correct = False
for i, prediction in enumerate(preds):
    is_match = np.array_equal(answer, prediction)
    if prediction.shape == answer.shape:
        match_pct = np.sum(prediction == answer) / answer.size * 100
        print(f"\nPrediction {i+1}: {match_pct:.1f}% match")
    else:
        print(f"\nPrediction {i+1}: Shape mismatch ({prediction.shape} vs {answer.shape})")
    print(f"Matches: {is_match}")
    if is_match:
        correct = True

print(f"\n{'='*60}")
print(f"FINAL RESULT: {'CORRECT' if correct else 'INCORRECT'}")
