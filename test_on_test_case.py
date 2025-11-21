import json
import os.path
import numpy as np

from ArcAgent import rule_fill_frames_with_frequent_color

# Load the problem
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Test on the test case
    test_data = flat_data['test'][0]
    test_inp = np.array(test_data['input'])
    test_out_expected = np.array(test_data['output'])

    predicted = rule_fill_frames_with_frequent_color(test_inp)

    match = np.array_equal(predicted, test_out_expected)

    print("=== Test Case ===")
    print(f"Match: {match}")

    if not match:
        if predicted.shape == test_out_expected.shape:
            diff_count = np.sum(predicted != test_out_expected)
            total_cells = predicted.size
            match_pct = (total_cells - diff_count) / total_cells * 100
            print(f"Shape: {predicted.shape}")
            print(f"Cells matching: {total_cells - diff_count}/{total_cells} ({match_pct:.1f}%)")

            # Show some differences
            diff_locations = np.where(predicted != test_out_expected)
            print(f"\nFirst 10 differences:")
            for i in range(min(10, len(diff_locations[0]))):
                r, c = diff_locations[0][i], diff_locations[1][i]
                print(f"  ({r},{c}): predicted={predicted[r,c]}, expected={test_out_expected[r,c]}, input={test_inp[r,c]}")
        else:
            print(f"Shape mismatch: predicted {predicted.shape} vs expected {test_out_expected.shape}")
