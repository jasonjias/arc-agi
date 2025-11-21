import json
import os.path
import numpy as np

from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcAgent import rule_fill_frames_with_frequent_color, np_equal

# Load a single problem to test
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Test on first training example
    print("Testing rule_fill_frames_with_frequent_color on training examples:")
    print("="*70)

    for i, dt in enumerate(flat_data['train']):
        inp = np.array(dt['input'])
        expected_out = np.array(dt['output'])

        predicted = rule_fill_frames_with_frequent_color(inp)

        match = np_equal(predicted, expected_out)

        print(f"\nTraining Example {i+1}:")
        print(f"  Match: {match}")

        if not match:
            # Show differences
            if predicted.shape == expected_out.shape:
                diff_count = np.sum(predicted != expected_out)
                total_cells = predicted.size
                match_pct = (total_cells - diff_count) / total_cells * 100
                print(f"  Shape matches: {predicted.shape}")
                print(f"  Cells matching: {total_cells - diff_count}/{total_cells} ({match_pct:.1f}%)")
            else:
                print(f"  Shape mismatch: predicted {predicted.shape} vs expected {expected_out.shape}")
