import json
import os.path
import numpy as np

from ArcAgent import detect_rectangular_frames

# Load a single problem to test
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Look at first training example
    dt = flat_data['train'][0]
    inp = np.array(dt['input'])
    expected_out = np.array(dt['output'])

    print("Input shape:", inp.shape)
    print("\nDetected frames:")
    frames = detect_rectangular_frames(inp)
    for frame_color, y0, y1, x0, x1 in frames:
        print(f"  Color {frame_color}: rows {y0}-{y1}, cols {x0}-{x1} (size: {y1-y0+1}x{x1-x0+1})")

        # Look at what's inside
        interior = inp[y0+1:y1, x0+1:x1]
        colors_inside = {}
        for val in interior.flatten():
            if val != 0 and val != frame_color:
                colors_inside[val] = colors_inside.get(val, 0) + 1

        print(f"    Interior colors: {colors_inside}")

        # What should be filled?
        expected_interior = expected_out[y0+1:y1, x0+1:x1]
        expected_fill = None
        for val in expected_interior.flatten():
            if val != 0 and val != frame_color:
                expected_fill = val
                break
        print(f"    Expected fill color: {expected_fill}")
