import json
import os.path
import numpy as np

# Inline the function with debug output
def rule_fit_shapes_into_holes_debug(inp: np.ndarray) -> np.ndarray:
    """Fit floating shapes into holes in the ground by trying all rotations."""
    h, w = inp.shape
    bg = 0
    out = inp.copy()

    # Find the solid ground base
    base_row = None
    ground_color = None

    if h > 0:
        last_row_colors = {}
        for val in inp[h-1, :]:
            if val != bg:
                last_row_colors[val] = last_row_colors.get(val, 0) + 1
        if last_row_colors:
            most_common = max(last_row_colors.items(), key=lambda x: x[1])
            if most_common[1] >= w // 2:
                base_row = h - 1
                ground_color = most_common[0]

    if base_row is None:
        return out

    # Find the row with holes
    holes_row = None
    if base_row > 0:
        row_above = base_row - 1
        has_ground_color = np.any(inp[row_above, :] == ground_color)
        has_background = np.any(inp[row_above, :] == bg)
        if has_ground_color and has_background:
            holes_row = row_above

    if holes_row is None:
        return out

    print(f"Base row: {base_row}, Holes row: {holes_row}")
    print(f"Holes row content: {inp[holes_row, :]}")

    # Find all floating shapes
    shapes = []
    visited = np.zeros((h, w), dtype=bool)

    for r in range(holes_row):
        for c in range(w):
            if not visited[r, c] and inp[r, c] != bg and inp[r, c] != ground_color:
                shape_color = inp[r, c]
                stack = [(r, c)]
                shape_cells = []
                visited[r, c] = True

                while stack:
                    rr, cc = stack.pop()
                    shape_cells.append((rr, cc))

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < holes_row and 0 <= nc < w:
                            if not visited[nr, nc] and inp[nr, nc] == shape_color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                if shape_cells:
                    shapes.append((shape_color, shape_cells))

    # Extract shape arrays
    shape_arrays = []
    for shape_color, cells in shapes:
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        shape_h = max_r - min_r + 1
        shape_w = max_c - min_c + 1
        shape_arr = np.zeros((shape_h, shape_w), dtype=inp.dtype)

        for r, c in cells:
            shape_arr[r - min_r, c - min_c] = shape_color

        shape_arrays.append((shape_color, shape_arr))

    print(f"\nFound {len(shape_arrays)} shapes:")
    for i, (sc, sa) in enumerate(shape_arrays):
        print(f"  Shape {i}: color {sc}, size {sa.shape}, cells {np.count_nonzero(sa)}")

    # Clear everything above holes_row
    out[:holes_row, :] = bg

    def get_rotations(arr):
        return [arr, np.rot90(arr), np.rot90(arr, 2), np.rot90(arr, 3)]

    # Sort shapes by area
    shape_arrays_with_idx = [(i, shape_color, shape_arr, np.count_nonzero(shape_arr))
                              for i, (shape_color, shape_arr) in enumerate(shape_arrays)]
    shape_arrays_with_idx.sort(key=lambda x: x[3])

    print(f"\nSorted order (by area):")
    for orig_idx, sc, sa, area in shape_arrays_with_idx:
        print(f"  Shape {orig_idx}: color {sc}, area {area}")

    placed_shapes = set()

    for orig_idx, shape_color, shape_arr, _ in shape_arrays_with_idx:
        print(f"\nTrying to place shape {orig_idx} (color {shape_color})...")
        if orig_idx in placed_shapes:
            continue

        placed = False
        for rot_idx, rotation in enumerate(get_rotations(shape_arr)):
            if placed:
                break
            sh, sw = rotation.shape
            print(f"  Rotation {rot_idx}: size {sh}x{sw}")

            for start_col in range(w - sw + 1):
                # Check if all columns needed are holes
                all_holes = True
                for sc in range(sw):
                    if inp[holes_row, start_col + sc] != bg:
                        all_holes = False
                        break

                if not all_holes:
                    continue

                print(f"    Column {start_col}: holes OK, checking placement...")

                if holes_row < sh:
                    print(f"      Not enough rows ({holes_row} < {sh})")
                    continue

                # Try to place
                can_place = True
                for sr in range(sh):
                    for sc in range(sw):
                        if rotation[sr, sc] != bg:
                            target_row = holes_row - sh + 1 + sr
                            target_col = start_col + sc
                            if target_row < 0 or target_row > holes_row:
                                can_place = False
                                break
                            if target_row == holes_row:
                                if inp[holes_row, target_col] != bg:
                                    can_place = False
                                    break
                            else:
                                if out[target_row, target_col] != bg:
                                    can_place = False
                                    break
                    if not can_place:
                        break

                if can_place:
                    print(f"      PLACING at column {start_col}!")
                    for sr in range(sh):
                        for sc in range(sw):
                            if rotation[sr, sc] != bg:
                                target_row = holes_row - sh + 1 + sr
                                target_col = start_col + sc
                                out[target_row, target_col] = rotation[sr, sc]

                    placed = True
                    placed_shapes.add(orig_idx)
                    break
                else:
                    print(f"      Cannot place (collision)")

        if not placed:
            print(f"  FAILED to place shape {orig_idx}")

    return out


# Test it
problem_name = "67c52801.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    train_input = np.array(flat_data['train'][0]['input'])
    expected_output = np.array(flat_data['train'][0]['output'])

    result = rule_fit_shapes_into_holes_debug(train_input)

    print("\n\nResult:")
    print(result)
    print("\nExpected:")
    print(expected_output)
    print(f"\nMatch: {np.array_equal(result, expected_output)}")
