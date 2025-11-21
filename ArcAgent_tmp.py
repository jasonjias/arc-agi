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


def flood_fill_region(grid: np.ndarray, start_r: int, start_c: int, target_val: int, visited: np.ndarray) -> List[Tuple[int, int]]:
    """Flood fill to find a connected region of target_val."""
    h, w = grid.shape
    if visited[start_r, start_c] or grid[start_r, start_c] != target_val:
        return []

    stack = [(start_r, start_c)]
    region = []
    visited[start_r, start_c] = True

    while stack:
        r, c = stack.pop()
        region.append((r, c))

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == target_val:
                visited[nr, nc] = True
                stack.append((nr, nc))

    return region


def flood_fill_non_frame(grid: np.ndarray, start_r: int, start_c: int, frame_color: int, visited: np.ndarray) -> List[Tuple[int, int]]:
    """Flood fill to find all cells that are NOT the frame color, connected via non-frame cells."""
    h, w = grid.shape
    if visited[start_r, start_c] or grid[start_r, start_c] == frame_color:
        return []

    stack = [(start_r, start_c)]
    region = []
    visited[start_r, start_c] = True

    while stack:
        r, c = stack.pop()
        region.append((r, c))

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] != frame_color:
                visited[nr, nc] = True
                stack.append((nr, nc))

    return region


def find_enclosed_regions_by_color(grid: np.ndarray, bg: int = 0) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """Find regions that are enclosed by non-background colors.
    Returns list of (enclosing_color, enclosed_cells) tuples.
    Enclosed cells include all non-frame cells in the enclosed area."""
    h, w = grid.shape
    enclosed_regions = []

    # Find all non-background colors (potential frame colors)
    frame_colors = [c for c in np.unique(grid) if c != bg]

    for frame_color in frame_colors:
        # Find all regions of non-frame cells
        visited = np.zeros((h, w), dtype=bool)

        for r in range(h):
            for c in range(w):
                if not visited[r, c] and grid[r, c] != frame_color:
                    # Flood fill to find all connected non-frame cells
                    region = flood_fill_non_frame(grid, r, c, frame_color, visited)

                    if region and len(region) > 0:
                        # Check if this region touches the border
                        touches_border = any(rr == 0 or rr == h-1 or cc == 0 or cc == w-1
                                            for rr, cc in region)

                        if not touches_border:
                            # This is an enclosed region
                            enclosed_regions.append((frame_color, region))

    return enclosed_regions


def rule_fill_frames_with_frequent_color(inp: np.ndarray) -> np.ndarray:
    """Fill enclosed regions with the most frequent non-background, non-frame color inside them."""
    bg = 0
    out = inp.copy()

    # Find regions enclosed by each color
    enclosed_regions = find_enclosed_regions_by_color(inp, bg=bg)

    for frame_color, region_cells in enclosed_regions:
        # Count colors in this enclosed region (excluding background and frame)
        color_counts: dict[int, int] = {}
        for r, c in region_cells:
            val = inp[r, c]
            if val != bg and val != frame_color:
                color_counts[val] = color_counts.get(val, 0) + 1

        if not color_counts:
            # No fill color found, skip
            continue

        # Get most frequent color
        fill_color = max(color_counts.keys(), key=lambda c: color_counts[c])

        # Fill the entire region with this color
        for r, c in region_cells:
            if out[r, c] != frame_color:  # Don't overwrite the frame itself
                out[r, c] = fill_color

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
    rule_fill_frames_with_frequent_color,
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
