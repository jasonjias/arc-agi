import json
import os.path
import numpy as np

from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcAgent import ArcAgent

# List of problems to test
problems = [
    "195ba7dc.json",  # Superimpose across colored divider
    "bbb1b8b6.json",  # Fuse across gray divider
    "4b6b68e5.json",  # Fill enclosed rings
    "60a26a3e.json",  # Connect flower petals
    "18419cfa.json",  # Reflect shapes in trays
    "d931c21c.json",  # Draw rings around blue rings
]

milestone_path = os.path.join('Milestones', 'D')
agent = ArcAgent()

results = []
print("="*80)
print("TESTING ALL MILESTONE D PROBLEMS")
print("="*80)

for problem_name in problems:
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
    print(f"\nTesting: {problem_name}")
    print("-"*80)
    preds = agent.make_predictions(arc_problem)

    # Check if correct
    answer = arc_problem.test_set().get_output_data().data()
    correct = False
    best_match = 0.0

    for i, prediction in enumerate(preds):
        is_match = np.array_equal(answer, prediction)
        if prediction.shape == answer.shape:
            match_pct = np.sum(prediction == answer) / answer.size * 100
            best_match = max(best_match, match_pct)
            print(f"Prediction {i+1}: {match_pct:.1f}% match")
        else:
            print(f"Prediction {i+1}: Shape mismatch ({prediction.shape} vs {answer.shape})")

        if is_match:
            correct = True
            print(f"✓ CORRECT on prediction {i+1}")

    result_status = "CORRECT ✓" if correct else f"INCORRECT (best: {best_match:.1f}%)"
    results.append((problem_name, correct, best_match))
    print(f"Result: {result_status}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
total_correct = sum(1 for _, correct, _ in results if correct)
print(f"Total Problems: {len(problems)}")
print(f"Correct: {total_correct}")
print(f"Success Rate: {total_correct}/{len(problems)} ({100*total_correct/len(problems):.1f}%)")
print()

for problem_name, correct, best_match in results:
    status = "✓" if correct else f"✗ ({best_match:.1f}%)"
    print(f"  {status} {problem_name}")

print("\n" + "="*80)
if total_correct >= 6:
    print("SUCCESS! You have solved at least 6 problems for Milestone D!")
else:
    print(f"Need {6 - total_correct} more problem(s) to meet the minimum requirement.")
print("="*80)
