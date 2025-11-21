import json
import os.path
import numpy as np

# Load the problem
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Look at first training example
    dt = flat_data['train'][0]
    inp = np.array(dt['input'])
    expected_out = np.array(dt['output'])

    print("=== First Training Example ===")
    print("\nInput (showing first frame - color 2):")
    # Find where color 2 appears
    ys, xs = np.where(inp == 2)
    if len(ys) > 0:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        print(f"Color 2 bounding box: rows {y0}-{y1}, cols {x0}-{x1}")
        print(f"Coordinates of color 2: {list(zip(ys[:20], xs[:20]))}")  # first 20

        # Check corners
        corners = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]
        print(f"Corners: {corners}")
        for corner in corners:
            print(f"  {corner}: color {inp[corner[0], corner[1]]}")

    print("\nColors in input:", np.unique(inp))
    print("Colors in output:", np.unique(expected_out))

    # Show a small region
    print("\nInput region (rows 1-8, cols 5-12):")
    print(inp[1:8, 5:12])
    print("\nExpected output region (rows 1-8, cols 5-12):")
    print(expected_out[1:8, 5:12])
