import numpy as np
import json
import sys
import os
sys.path.insert(0, '.')
from ArcAgent_combined import ArcAgent
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcData import ArcData


# Define all problems by milestone
MILESTONE_B_PROBLEMS = [
    "0520fde7", "1cf80156", "28e73c20", "4347f46a", "6150a2bd", "623ea044",
    "6430c8c4", "b1948b0a", "b94a9452", "ce22a75a", "ce4f8723", "d687bc17",
    "ed36ccf7", "f25ffba3", "f2829549", "f76d97a5"
]

MILESTONE_C_PROBLEMS = [
    "22eb0ac0", "25d487eb", "3428a4f5", "3de23699", "5c0a986e", "62c24649",
    "74dd1130", "7b6016b9", "9af7a82c", "b2862040", "bbc9ae5d", "cf98881b",
    "dc433765", "e98196ab", "f35d900a", "f8a8fe49"
]

MILESTONE_D_PROBLEMS = [
    "18419cfa", "195ba7dc", "2546ccf6", "31d5ba1a", "4b6b68e5", "60a26a3e",
    "67c52801", "81c0276b", "992798f6", "bbb1b8b6", "bcb3040b", "c1990cce",
    "c48954c1", "c8b7cc0f", "d931c21c", "e9b4f6fc"
]


def test_problem(problem_id, milestone):
    json_path = f'./Milestones/{milestone}/{problem_id}.json'

    if not os.path.exists(json_path):
        return False, f"File not found: {json_path}"

    try:
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

        for i, pred in enumerate(predictions, 1):
            if pred.shape == expected.shape:
                if np.array_equal(pred, expected):
                    return True, f"PASS (prediction {i})"

        return False, "FAIL - No exact match"

    except Exception as e:
        return False, f"ERROR: {str(e)}"


def main():
    print("=" * 80)
    print("ARC-AGI Combined Agent Test Suite")
    print("=" * 80)

    all_results = []

    # Test Milestone B
    print("\n" + "=" * 80)
    print("MILESTONE B")
    print("=" * 80)
    b_passed = 0
    for problem_id in MILESTONE_B_PROBLEMS:
        success, message = test_problem(problem_id, 'B')
        status = "✓" if success else "✗"
        print(f"{status} {problem_id}: {message}")
        if success:
            b_passed += 1
        all_results.append((problem_id, 'B', success))
    print(f"\nMilestone B: {b_passed}/{len(MILESTONE_B_PROBLEMS)} passed")

    # Test Milestone C
    print("\n" + "=" * 80)
    print("MILESTONE C")
    print("=" * 80)
    c_passed = 0
    for problem_id in MILESTONE_C_PROBLEMS:
        success, message = test_problem(problem_id, 'C')
        status = "✓" if success else "✗"
        print(f"{status} {problem_id}: {message}")
        if success:
            c_passed += 1
        all_results.append((problem_id, 'C', success))
    print(f"\nMilestone C: {c_passed}/{len(MILESTONE_C_PROBLEMS)} passed")

    # Test Milestone D
    print("\n" + "=" * 80)
    print("MILESTONE D")
    print("=" * 80)
    d_passed = 0
    for problem_id in MILESTONE_D_PROBLEMS:
        success, message = test_problem(problem_id, 'D')
        status = "✓" if success else "✗"
        print(f"{status} {problem_id}: {message}")
        if success:
            d_passed += 1
        all_results.append((problem_id, 'D', success))
    print(f"\nMilestone D: {d_passed}/{len(MILESTONE_D_PROBLEMS)} passed")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_passed = b_passed + c_passed + d_passed
    total_problems = len(MILESTONE_B_PROBLEMS) + len(MILESTONE_C_PROBLEMS) + len(MILESTONE_D_PROBLEMS)
    print(f"Milestone B: {b_passed}/{len(MILESTONE_B_PROBLEMS)} passed")
    print(f"Milestone C: {c_passed}/{len(MILESTONE_C_PROBLEMS)} passed")
    print(f"Milestone D: {d_passed}/{len(MILESTONE_D_PROBLEMS)} passed")
    print(f"\nTotal: {total_passed}/{total_problems} passed ({100.0 * total_passed / total_problems:.1f}%)")
    print("=" * 80)

    return 0 if total_passed == total_problems else 1


if __name__ == "__main__":
    sys.exit(main())
