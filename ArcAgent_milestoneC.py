import numpy as np
from typing import Callable, List, Tuple
from ArcProblem import ArcProblem
from ArcSet import ArcSet


def np_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def hstack_or_pad(blocks: List[np.ndarray], pad_val: int = 0) -> np.ndarray:
    if not blocks:
        return np.zeros((1, 1), dtype=int)
    max_h = max(b.shape[0] for b in blocks)
    rows = []
    for b in blocks:
        h, w = b.shape
        canvas = np.full((max_h, w), pad_val, dtype=int)
        canvas[:h, :w] = b
        rows.append(canvas)
    return np.concatenate(rows, axis=1)


def bounding_box_of_foreground(grid: np.ndarray, bg: int = 0) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(grid != bg)
    if ys.size == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def extract_connected_components(grid: np.ndarray, bg: int = 0) -> List[np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []

    def neighbors(r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc

    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            color = grid[r, c]
            if color == bg:
                continue
            stack = [(r, c)]
            cells = []
            visited[r, c] = True
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors(rr, cc):
                    if not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            comp = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=int)
            for yy, xx in cells:
                comp[yy - y0, xx - x0] = grid[yy, xx]
            comps.append(comp)
    return comps


def rule_identity(inp: np.ndarray) -> np.ndarray:
    return inp.copy()


def rule_bounding_box_fill(inp: np.ndarray) -> np.ndarray:
    bg = 0
    bb = bounding_box_of_foreground(inp, bg=bg)
    if bb is None:
        return np.zeros_like(inp)
    y0, y1, x0, x1 = bb
    out = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=int)
    out[:, :] = inp[y0 : y1 + 1, x0 : x1 + 1]
    return out


def rule_tile2x2(inp: np.ndarray) -> np.ndarray:
    A = inp
    Ax = np.fliplr(A)
    Ay = np.flipud(A)
    Axy = np.flipud(np.fliplr(A))
    top = np.concatenate([A, Ax], axis=1)
    bot = np.concatenate([Ay, Axy], axis=1)
    return np.concatenate([top, bot], axis=0)


def rule_components_row(inp: np.ndarray) -> np.ndarray:
    comps = extract_connected_components(inp, bg=0)
    comps.sort(key=lambda g: (-np.count_nonzero(g), int(g[g != 0][0]) if np.any(g != 0) else -1))
    return hstack_or_pad(comps, pad_val=0)


def rule_merge_across_gray_line(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    gray_row = None
    for y in range(h):
        row = inp[y, :]
        if len(np.unique(row)) == 1 and row[0] != 0:
            gray_row = y
            break
    if gray_row is None:
        return inp.copy()
    top = inp[:gray_row, :]
    bot = inp[gray_row + 1 :, :]
    hh = max(top.shape[0], bot.shape[0])

    def pad(arr):
        pad_h = hh - arr.shape[0]
        if pad_h > 0:
            return np.pad(arr, ((pad_h, 0), (0, 0)), constant_values=0)
        return arr

    top_p = pad(top)
    bot_p = pad(bot)
    merged = np.where(top_p != 0, top_p, bot_p)
    return merged


def rule_transpose(inp: np.ndarray) -> np.ndarray:
    return inp.T.copy()


def rule_anchor_frame_extract(inp: np.ndarray) -> np.ndarray:
    bg = 0
    colors = [c for c in np.unique(inp) if c != bg]

    def bbox_of_color(color: int):
        ys, xs = np.where(inp == color)
        if ys.size == 0:
            return None
        return ys.min(), ys.max(), xs.min(), xs.max()

    def looks_like_anchor(color: int, y0, y1, x0, x1) -> bool:
        coords = set(zip(*np.where(inp == color)))
        corners = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]
        return all(corner in coords for corner in corners)

    anchor_color = None
    anchor_box = None
    for c in colors:
        bb = bbox_of_color(c)
        if bb is None:
            continue
        y0, y1, x0, x1 = bb
        if (y1 - y0) >= 2 and (x1 - x0) >= 2:
            if looks_like_anchor(c, y0, y1, x0, x1):
                anchor_color = c
                anchor_box = (y0, y1, x0, x1)
                break
    if anchor_color is None or anchor_box is None:
        return inp.copy()
    y0, y1, x0, x1 = anchor_box
    inner = inp[y0 + 1 : y1, x0 + 1 : x1]
    if inner.size == 0:
        return np.zeros((0, 0), dtype=int)
    out = np.zeros_like(inner)
    mask = inner != bg
    out[mask] = anchor_color
    return out


def find_solid_blocks(inp: np.ndarray, block_size: int = 2) -> List[tuple]:
    h, w = inp.shape
    hits = []
    for r in range(h - block_size + 1):
        for c in range(w - block_size + 1):
            patch = inp[r : r + block_size, c : c + block_size]
            vals = patch.flatten()
            if np.all(vals != 0) and np.all(vals == vals[0]):
                hits.append((int(vals[0]), r, c))
    return hits


def infer_direction_mapping(train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for tin, tout in train_pairs:
        h, w = tin.shape
        blocks = find_solid_blocks(tin, block_size=2)
        for color, r0, c0 in blocks:
            if color in mapping:
                continue
            up_anchor = (r0, c0)
            dn_anchor = (r0 + 1, c0 + 1)

            def trace_cells(start_r, start_c, dr, dc):
                rr, cc = start_r, start_c
                cells = []
                while 0 <= rr < h and 0 <= cc < w:
                    cells.append((rr, cc))
                    rr += dr
                    cc += dc
                return cells

            up_path = trace_cells(up_anchor[0], up_anchor[1], -1, -1)
            dn_path = trace_cells(dn_anchor[0], dn_anchor[1], +1, +1)
            up_score = sum(tout[r, c] == color for (r, c) in up_path)
            dn_score = sum(tout[r, c] == color for (r, c) in dn_path)
            if up_score > dn_score and up_score > 0:
                mapping[color] = "upleft"
            elif dn_score > up_score and dn_score > 0:
                mapping[color] = "downright"
    return mapping


def apply_trails(inp: np.ndarray, mapping: dict[int, str]) -> np.ndarray:
    h, w = inp.shape
    out = np.zeros_like(inp)
    blocks = find_solid_blocks(inp, block_size=2)
    for color, r0, c0 in blocks:
        direction = mapping.get(color)
        if direction is None:
            continue
        if direction == "upleft":
            rr, cc = r0, c0
            while rr >= 0 and cc >= 0:
                out[rr, cc] = color
                rr -= 1
                cc -= 1
        elif direction == "downright":
            rr, cc = r0 + 1, c0 + 1
            while rr < h and cc < w:
                out[rr, cc] = color
                rr += 1
                cc += 1
        out[r0 : r0 + 2, c0 : c0 + 2] = color
    return out


def rule_diagonal_trails_from_blocks(inp: np.ndarray, train_pairs: List[Tuple[np.ndarray, np.ndarray]] | None = None) -> np.ndarray:
    if train_pairs is None or len(train_pairs) == 0:
        return inp.copy()
    mapping = infer_direction_mapping(train_pairs)
    return apply_trails(inp, mapping)


def rule_fill_rows_by_matching_edges(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    out = inp.copy()
    for r in range(h):
        left_color = inp[r, 0]
        right_color = inp[r, w - 1]
        if left_color != 0 and left_color == right_color:
            out[r, :] = left_color
    return out


def rule_color_frequency_columns(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    bg = 0
    counts: dict[int, int] = {}
    first_pos: dict[int, Tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(inp[r, c])
            if color == bg:
                continue
            counts[color] = counts.get(color, 0) + 1
            if color not in first_pos:
                first_pos[color] = (r, c)
    if not counts:
        return np.zeros((1, 1), dtype=int)
    colors_sorted = sorted(counts.keys(), key=lambda col: (-counts[col], first_pos[col]))
    max_h = max(counts[c] for c in colors_sorted)
    out_w = len(colors_sorted)
    out = np.zeros((max_h, out_w), dtype=int)
    for col_idx, color in enumerate(colors_sorted):
        freq = counts[color]
        out[:freq, col_idx] = color
    return out


def find_enclosed_regions(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    bg = 0
    grid_color = None

    color_counts = {}
    for val in inp.flat:
        if val != bg:
            color_counts[val] = color_counts.get(val, 0) + 1

    if color_counts:
        grid_color = max(color_counts.items(), key=lambda x: x[1])[0]

    if grid_color is None:
        return inp.copy()

    out = np.full_like(inp, 3)
    out[inp == grid_color] = grid_color

    visited = np.zeros((h, w), dtype=bool)

    def flood_fill(start_r, start_c):
        if visited[start_r, start_c] or inp[start_r, start_c] == grid_color:
            return None

        stack = [(start_r, start_c)]
        region_cells = []
        touches_border = False
        visited[start_r, start_c] = True

        while stack:
            r, c = stack.pop()
            region_cells.append((r, c))

            if r == 0 or r == h-1 or c == 0 or c == w-1:
                touches_border = True

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if not visited[nr, nc] and inp[nr, nc] != grid_color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

        return (region_cells, touches_border)

    for r in range(h):
        for c in range(w):
            if not visited[r, c] and inp[r, c] != grid_color:
                result = flood_fill(r, c)
                if result:
                    region_cells, touches_border = result
                    fill_color = 3 if touches_border else 2
                    for rr, cc in region_cells:
                        out[rr, cc] = fill_color

    return out


def rule_fill_enclosed_regions_with_red_rest_green(inp: np.ndarray) -> np.ndarray:
    return find_enclosed_regions(inp)


def rule_spaceship_laser(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    out = inp.copy()
    bg = 0

    color_counts = {}
    for val in inp.flat:
        if val != bg:
            color_counts[val] = color_counts.get(val, 0) + 1

    if len(color_counts) < 2:
        return out

    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
    laser_color = sorted_colors[0][0]
    ship_color = sorted_colors[1][0]

    laser_pos = None
    for r in range(h):
        for c in range(w):
            if inp[r, c] == laser_color:
                laser_pos = (r, c)
                break
        if laser_pos:
            break

    if laser_pos is None:
        return out

    laser_r, laser_c = laser_pos

    ship_coords = []
    for r in range(h):
        for c in range(w):
            if inp[r, c] == ship_color:
                ship_coords.append((r, c))

    if not ship_coords:
        return out

    ship_rows = [r for r, c in ship_coords]
    ship_cols = [c for r, c in ship_coords]
    ship_center_r = sum(ship_rows) / len(ship_rows)
    ship_center_c = sum(ship_cols) / len(ship_cols)

    dr = ship_center_r - laser_r
    dc = ship_center_c - laser_c

    if abs(dr) > abs(dc):
        if dr > 0:
            for r in range(laser_r + 1, h):
                if out[r, laser_c] == bg:
                    out[r, laser_c] = laser_color
        else:
            for r in range(laser_r - 1, -1, -1):
                if out[r, laser_c] == bg:
                    out[r, laser_c] = laser_color
    else:
        if dc > 0:
            for c in range(laser_c + 1, w):
                if out[laser_r, c] == bg:
                    out[laser_r, c] = laser_color
        else:
            for c in range(laser_c - 1, -1, -1):
                if out[laser_r, c] == bg:
                    out[laser_r, c] = laser_color

    return out


def find_shapes_with_holes(inp: np.ndarray) -> List[set]:
    h, w = inp.shape
    bg = 9

    shape_color = None
    for val in inp.flat:
        if val != bg:
            shape_color = val
            break

    if shape_color is None:
        return []

    visited = np.zeros((h, w), dtype=bool)
    shapes_with_holes = []

    def flood_fill_shape(start_r, start_c):
        if visited[start_r, start_c] or inp[start_r, start_c] != shape_color:
            return None

        stack = [(start_r, start_c)]
        shape_cells = set()
        visited[start_r, start_c] = True

        while stack:
            r, c = stack.pop()
            shape_cells.add((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if not visited[nr, nc] and inp[nr, nc] == shape_color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

        return shape_cells

    for r in range(h):
        for c in range(w):
            if not visited[r, c] and inp[r, c] == shape_color:
                shape_cells = flood_fill_shape(r, c)
                if shape_cells:
                    rows = [rr for rr, cc in shape_cells]
                    cols = [cc for rr, cc in shape_cells]
                    min_r, max_r = min(rows), max(rows)
                    min_c, max_c = min(cols), max(cols)

                    visited_hole = np.zeros((h, w), dtype=bool)
                    for rr, cc in shape_cells:
                        visited_hole[rr, cc] = True

                    has_hole = False
                    for rr in range(min_r, max_r + 1):
                        for cc in range(min_c, max_c + 1):
                            if inp[rr, cc] == bg and not visited_hole[rr, cc]:
                                hole_stack = [(rr, cc)]
                                hole_cells = []
                                visited_hole[rr, cc] = True

                                while hole_stack:
                                    hr, hc = hole_stack.pop()
                                    hole_cells.append((hr, hc))

                                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                        nhr, nhc = hr + dr, hc + dc
                                        if 0 <= nhr < h and 0 <= nhc < w:
                                            if not visited_hole[nhr, nhc] and inp[nhr, nhc] == bg:
                                                visited_hole[nhr, nhc] = True
                                                hole_stack.append((nhr, nhc))

                                touches_shape_border = False
                                for hr, hc in hole_cells:
                                    if hr == 0 or hr == h-1 or hc == 0 or hc == w-1:
                                        touches_shape_border = True
                                        break

                                if not touches_shape_border:
                                    has_hole = True
                                    break

                        if has_hole:
                            break

                    if has_hole:
                        shapes_with_holes.append(shape_cells)

    return shapes_with_holes


def rule_recolor_shapes_with_holes(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    shapes_with_holes = find_shapes_with_holes(inp)

    for shape_cells in shapes_with_holes:
        for r, c in shape_cells:
            out[r, c] = 8

    return out


def rule_move_green_toward_yellow(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    out = inp.copy()

    green = 3
    yellow = 4

    green_pos = None
    yellow_pos = None

    for r in range(h):
        for c in range(w):
            if inp[r, c] == green:
                green_pos = (r, c)
            if inp[r, c] == yellow:
                yellow_pos = (r, c)

    if green_pos is None or yellow_pos is None:
        return out

    gr, gc = green_pos
    yr, yc = yellow_pos

    dr = yr - gr
    dc = yc - gc

    move_r = 0
    move_c = 0

    if dr > 0:
        move_r = 1
    elif dr < 0:
        move_r = -1

    if dc > 0:
        move_c = 1
    elif dc < 0:
        move_c = -1

    new_gr = gr + move_r
    new_gc = gc + move_c

    out[gr, gc] = 0
    out[new_gr, new_gc] = green

    return out


def rule_reflect_shapes_outside_brackets(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    out = inp.copy()

    bracket_color = 2
    shape_color = 5
    bg = 0

    bracket_coords = set()
    for r in range(h):
        for c in range(w):
            if inp[r, c] == bracket_color:
                bracket_coords.add((r, c))

    if not bracket_coords:
        return out

    bracket_rows = [r for r, c in bracket_coords]
    bracket_cols = [c for r, c in bracket_coords]
    min_br, max_br = min(bracket_rows), max(bracket_rows)
    min_bc, max_bc = min(bracket_cols), max(bracket_cols)

    visited = np.zeros((h, w), dtype=bool)
    shapes = []

    def flood_fill_shape(start_r, start_c):
        if visited[start_r, start_c] or inp[start_r, start_c] != shape_color:
            return None

        stack = [(start_r, start_c)]
        shape_cells = []
        visited[start_r, start_c] = True

        while stack:
            r, c = stack.pop()
            shape_cells.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if not visited[nr, nc] and inp[nr, nc] == shape_color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

        return shape_cells

    for r in range(h):
        for c in range(w):
            if not visited[r, c] and inp[r, c] == shape_color:
                shape_cells = flood_fill_shape(r, c)
                if shape_cells:
                    shapes.append(shape_cells)

    for shape_cells in shapes:
        for r, c in shape_cells:
            out[r, c] = 0

    for shape_cells in shapes:
        shape_rows = [r for r, c in shape_cells]
        shape_cols = [c for r, c in shape_cells]
        center_r = sum(shape_rows) / len(shape_rows)
        center_c = sum(shape_cols) / len(shape_cols)

        dist_to_top = center_r - min_br
        dist_to_bottom = max_br - center_r
        dist_to_left = center_c - min_bc
        dist_to_right = max_bc - center_c

        min_dist = min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right)

        if min_dist == dist_to_top:
            for r, c in shape_cells:
                new_r = min_br - (r - min_br)
                if 0 <= new_r < h:
                    out[new_r, c] = shape_color
        elif min_dist == dist_to_bottom:
            for r, c in shape_cells:
                new_r = max_br + (max_br - r)
                if 0 <= new_r < h:
                    out[new_r, c] = shape_color
        elif min_dist == dist_to_left:
            for r, c in shape_cells:
                new_c = min_bc - (c - min_bc)
                if 0 <= new_c < w:
                    out[r, new_c] = shape_color
        else:
            for r, c in shape_cells:
                new_c = max_bc + (max_bc - c)
                if 0 <= new_c < w:
                    out[r, new_c] = shape_color

    return out


def rule_stack_sections_divided_by_red(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    red = 2

    red_cols = []
    for c in range(w):
        if all(inp[r, c] == red for r in range(h)):
            red_cols.append(c)

    if len(red_cols) != 2:
        return inp.copy()

    left_section = inp[:, :red_cols[0]]
    middle_section = inp[:, red_cols[0]+1:red_cols[1]]
    right_section = inp[:, red_cols[1]+1:]

    section_width = min(left_section.shape[1], middle_section.shape[1], right_section.shape[1])

    out = np.zeros((h, section_width), dtype=inp.dtype)

    for r in range(h):
        for c in range(section_width):
            if left_section[r, c] != 0:
                out[r, c] = left_section[r, c]
            elif middle_section[r, c] != 0:
                out[r, c] = middle_section[r, c]
            elif right_section[r, c] != 0:
                out[r, c] = right_section[r, c]

    return out


def rule_xor_sections_divided_by_yellow(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    yellow = 4

    yellow_row = None
    for r in range(h):
        if all(inp[r, c] == yellow for c in range(w)):
            yellow_row = r
            break

    if yellow_row is None:
        return inp.copy()

    top_section = inp[:yellow_row, :]
    bottom_section = inp[yellow_row + 1:, :]

    section_height = min(top_section.shape[0], bottom_section.shape[0])

    out = np.zeros((section_height, w), dtype=inp.dtype)

    for r in range(section_height):
        for c in range(w):
            top_val = top_section[r, c]
            bottom_val = bottom_section[r, c]

            if (top_val != 0 and bottom_val == 0) or (top_val == 0 and bottom_val != 0):
                out[r, c] = 3
            else:
                out[r, c] = 0

    return out


def rule_staircase_from_starting_pixels(inp: np.ndarray) -> np.ndarray:
    if inp.shape[0] != 1:
        return inp.copy()

    row = inp[0, :]
    w = len(row)
    bg = 0

    # Count starting colored pixels
    n_starting = 0
    color = None
    for c in range(w):
        if row[c] != bg:
            n_starting += 1
            if color is None:
                color = row[c]
        else:
            break

    if n_starting == 0 or color is None:
        return inp.copy()

    # Calculate target: midpoint + (n_starting - 1)
    midpoint = w // 2
    target = midpoint + (n_starting - 1)

    if target > w:
        target = w

    num_rows = target - n_starting + 1

    if num_rows <= 0:
        return inp.copy()

    # Build staircase
    out = []
    for row_idx in range(num_rows):
        new_row = np.zeros(w, dtype=inp.dtype)
        fill_count = n_starting + row_idx
        if fill_count > w:
            fill_count = w
        new_row[:fill_count] = color
        out.append(new_row)

    return np.array(out, dtype=inp.dtype)


def rule_draw_boxes_and_connect_pixels(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    bg = 0
    out = inp.copy()

    # Find all non-background pixels
    pixels = []
    for r in range(h):
        for c in range(w):
            if inp[r, c] != bg:
                pixels.append((r, c, int(inp[r, c])))

    # Should have exactly 4 pixels
    if len(pixels) != 4:
        return inp.copy()

    # Group pixels by color
    color_groups = {}
    for r, c, color in pixels:
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append((r, c))

    # Should have exactly 2 colors, each with 2 pixels
    if len(color_groups) != 2:
        return inp.copy()

    colors = list(color_groups.keys())
    if len(color_groups[colors[0]]) != 2 or len(color_groups[colors[1]]) != 2:
        return inp.copy()

    color1, color2 = colors[0], colors[1]

    # Draw 3x3 boxes around each pixel with the OTHER color
    for r, c, pixel_color in pixels:
        box_color = color2 if pixel_color == color1 else color1

        # Draw 3x3 box
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if out[rr, cc] == bg:  # Don't overwrite existing pixels
                        out[rr, cc] = box_color

    # Draw dotted connecting lines (color 5) between pixel pairs
    dotted_color = 5

    # Find pairs of pixels on same row or column (regardless of color)
    for i in range(len(pixels)):
        for j in range(i + 1, len(pixels)):
            r1, c1, color1 = pixels[i]
            r2, c2, color2 = pixels[j]

            # Same column - vertical connection
            if c1 == c2:
                min_r = min(r1, r2)
                max_r = max(r1, r2)

                # Place dots near both edges
                near_edge = min_r + 2
                far_edge = max_r - 2

                # Place 2 dots near the top (starting from near_edge, every 2)
                if near_edge <= far_edge and out[near_edge, c1] == bg:
                    out[near_edge, c1] = dotted_color
                if near_edge + 2 <= far_edge and out[near_edge + 2, c1] == bg:
                    out[near_edge + 2, c1] = dotted_color

                # Place 2 dots near the bottom (ending at far_edge, every 2 backward)
                if far_edge > near_edge + 2 and out[far_edge, c1] == bg:
                    out[far_edge, c1] = dotted_color
                if far_edge - 2 > near_edge + 2 and out[far_edge - 2, c1] == bg:
                    out[far_edge - 2, c1] = dotted_color

            # Same row - horizontal connection
            if r1 == r2:
                min_c = min(c1, c2)
                max_c = max(c1, c2)

                # Place dots near both edges
                near_edge = min_c + 2
                far_edge = max_c - 2

                # Place 2 dots near the left (starting from near_edge, every 2)
                if near_edge <= far_edge and out[r1, near_edge] == bg:
                    out[r1, near_edge] = dotted_color
                if near_edge + 2 <= far_edge and out[r1, near_edge + 2] == bg:
                    out[r1, near_edge + 2] = dotted_color

                # Place 2 dots near the right (ending at far_edge, every 2 backward)
                if far_edge > near_edge + 2 and out[r1, far_edge] == bg:
                    out[r1, far_edge] = dotted_color
                if far_edge - 2 > near_edge + 2 and out[r1, far_edge - 2] == bg:
                    out[r1, far_edge - 2] = dotted_color

    return out


class DiagonalTrailRule:
    def __init__(self, train_pairs):
        self.train_pairs = train_pairs

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return rule_diagonal_trails_from_blocks(inp, self.train_pairs)


BASE_RULES: List[Callable[[np.ndarray], np.ndarray]] = [
    rule_identity,
    rule_transpose,
    rule_bounding_box_fill,
    rule_tile2x2,
    rule_components_row,
    rule_merge_across_gray_line,
    rule_anchor_frame_extract,
    rule_fill_rows_by_matching_edges,
    rule_color_frequency_columns,
    rule_fill_enclosed_regions_with_red_rest_green,
    rule_spaceship_laser,
    rule_recolor_shapes_with_holes,
    rule_move_green_toward_yellow,
    rule_reflect_shapes_outside_brackets,
    rule_stack_sections_divided_by_red,
    rule_xor_sections_divided_by_yellow,
    rule_staircase_from_starting_pixels,
    rule_draw_boxes_and_connect_pixels,
]


def score_rule_on_pairs(rule_fn: Callable[[np.ndarray], np.ndarray], train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    total = 0.0
    for tin, tout in train_pairs:
        pred = rule_fn(tin)
        if np_equal(pred, tout):
            total += 1.0
        else:
            if pred.shape == tout.shape:
                matches = (pred == tout).sum()
                total += matches / pred.size
            else:
                total += 0.0
    return total / max(len(train_pairs), 1)


class ArcAgent:
    def __init__(self) -> None:
        pass

    def _get_training_pairs(self, arc_problem: ArcProblem) -> List[Tuple[np.ndarray, np.ndarray]]:
        pairs = []
        for trn_set in arc_problem.training_set():
            tin = trn_set.get_input_data().data()
            tout = trn_set.get_output_data().data()
            pairs.append((tin, tout))
        return pairs

    def _get_test_input(self, arc_problem: ArcProblem) -> np.ndarray:
        return arc_problem.test_set().get_input_data().data()

    def make_predictions(self, arc_problem: ArcProblem) -> List[np.ndarray]:
        train_pairs = self._get_training_pairs(arc_problem)
        test_in = self._get_test_input(arc_problem)
        if len(train_pairs) == 0:
            return [test_in.copy()]
        dynamic_rules: List[Callable[[np.ndarray], np.ndarray]] = []
        dynamic_rules.extend(BASE_RULES)
        dynamic_rules.append(DiagonalTrailRule(train_pairs))
        scored_rules = []
        for rfn in dynamic_rules:
            try:
                sc = score_rule_on_pairs(rfn, train_pairs)
            except Exception:
                sc = -1.0
            scored_rules.append((sc, rfn))
        scored_rules.sort(key=lambda x: x[0], reverse=True)
        preds = []
        used = set()
        for sc, rfn in scored_rules:
            try:
                guess = rfn(test_in)
                sig = (guess.shape, guess.tobytes())
                if sig not in used:
                    preds.append(guess.astype(int))
                    used.add(sig)
            except Exception:
                continue
            if len(preds) >= 3:
                break
        if len(preds) == 0:
            preds = [test_in.copy()]
        return preds[:3]
