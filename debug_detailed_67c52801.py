import json
import os.path
import numpy as np

# Load the problem
problem_name = "67c52801.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)

    # Test on first training example
    train_input = np.array(flat_data['train'][0]['input'])
    expected_output = np.array(flat_data['train'][0]['output'])

    print("Training Example 1:")
    print("Input:")
    print(train_input)

    h, w = train_input.shape
    bg = 0

    # Find ground row
    print(f"\nLast row (row {h-1}): {train_input[h-1, :]}")
    last_row_colors = {}
    for val in train_input[h-1, :]:
        if val != bg:
            last_row_colors[val] = last_row_colors.get(val, 0) + 1
    print(f"Last row colors: {last_row_colors}")

    most_common = max(last_row_colors.items(), key=lambda x: x[1])
    print(f"Most common: {most_common}")
    ground_row = h - 1
    ground_color = most_common[0]
    print(f"Ground row: {ground_row}, Ground color: {ground_color}")

    # Find holes
    print(f"\nGround row content: {train_input[ground_row, :]}")
    holes = []
    for c in range(w):
        if train_input[ground_row, c] == bg:
            holes.append(c)
    print(f"Holes at columns: {holes}")

    # Find shapes
    print("\nFinding shapes:")
    shapes = []
    visited = np.zeros((h, w), dtype=bool)

    for r in range(ground_row):
        for c in range(w):
            if not visited[r, c] and train_input[r, c] != bg and train_input[r, c] != ground_color:
                shape_color = train_input[r, c]
                stack = [(r, c)]
                shape_cells = []
                visited[r, c] = True

                while stack:
                    rr, cc = stack.pop()
                    shape_cells.append((rr, cc))

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < ground_row and 0 <= nc < w:
                            if not visited[nr, nc] and train_input[nr, nc] == shape_color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                if shape_cells:
                    shapes.append((shape_color, shape_cells))
                    print(f"  Shape {len(shapes)}: color {shape_color}, cells {shape_cells}")

    # Extract shape arrays
    print("\nShape arrays:")
    shape_arrays = []
    for shape_color, cells in shapes:
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        shape_h = max_r - min_r + 1
        shape_w = max_c - min_c + 1
        shape_arr = np.zeros((shape_h, shape_w), dtype=train_input.dtype)

        for r, c in cells:
            shape_arr[r - min_r, c - min_c] = shape_color

        shape_arrays.append((shape_color, shape_arr))
        print(f"  Shape color {shape_color}, size {shape_h}x{shape_w}:")
        print(shape_arr)

    # Now test placement
    print("\n\nTesting placement:")
    print(f"Ground row: {ground_row}")

    for shape_idx, (shape_color, shape_arr) in enumerate(shape_arrays):
        print(f"\nShape {shape_idx} (color {shape_color}, size {shape_arr.shape}):")
        sh, sw = shape_arr.shape

        # Try each starting column
        for start_col in range(w - sw + 1):
            # Check if all columns needed are holes
            all_holes = True
            for sc in range(sw):
                if train_input[ground_row, start_col + sc] != bg:
                    all_holes = False
                    break

            print(f"  start_col={start_col}, sw={sw}, cols {start_col} to {start_col+sw-1}: all_holes={all_holes}")

            if all_holes:
                print(f"    Can try to place here!")
                print(f"    Need {sh} rows, ground_row={ground_row}, have room: {ground_row >= sh}")

                if ground_row >= sh:
                    print(f"    Would place at rows {ground_row - sh} to {ground_row - 1}, cols {start_col} to {start_col + sw - 1}")
