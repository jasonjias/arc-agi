import numpy as np
from typing import List, Tuple, Callable, Dict
from ArcProblem import ArcProblem
from ArcSet import ArcSet

class ArcAgentB:
    def __init__(self):
        pass
    def _mode_color(self, grid: np.ndarray):
        values, counts = np.unique(grid, return_counts=True)
        return int(values[np.argmax(counts)])
    def _crop_bounding_box_nonbg(self, grid: np.ndarray) -> np.ndarray:
        background = self._mode_color(grid)
        ys, xs = np.where(grid != background)
        if ys.size == 0:
            return grid.copy()
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        return grid[y0:y1, x0:x1]
    def _mirror_fill_left_right(self, grid: np.ndarray) -> np.ndarray:
        _, width = grid.shape
        mid = width // 2
        result = grid.copy()
        left_half = result[:, :mid]
        right_half = result[:, width - mid:]
        background = self._mode_color(result)
        left_count = np.count_nonzero(left_half != background)
        right_count = np.count_nonzero(right_half != background)
        if right_count == 0 and left_count > 0:
            result[:, width - mid:] = np.fliplr(left_half)
        elif left_count == 0 and right_count > 0:
            result[:, :mid] = np.fliplr(right_half)
        else:
            if left_count >= right_count:
                result[:, width - mid:] = np.fliplr(left_half)
            else:
                result[:, :mid] = np.fliplr(right_half)
        return result
    def _mirror_fill_up_down(self, grid):
        height, _ = grid.shape
        mid = height // 2
        result = grid.copy()
        top_half = result[:mid, :]
        bottom_half = result[height - mid:, :]
        background = self._mode_color(result)
        top_count = np.count_nonzero(top_half != background)
        bottom_count = np.count_nonzero(bottom_half != background)
        if bottom_count == 0 and top_count > 0:
            result[height - mid:, :] = np.flipud(top_half)
        elif top_count == 0 and bottom_count > 0:
            result[:mid, :] = np.flipud(bottom_half)
        else:
            if top_count >= bottom_count:
                result[height - mid:, :] = np.flipud(top_half)
            else:
                result[:mid, :] = np.flipud(bottom_half)
        return result
    def _apply_color_map(
        self, grid, color_map
    ):
        result = grid.copy()
        flat = result.ravel()
        for i, value in enumerate(flat):
            flat[i] = color_map.get(int(value), int(value))
        return result
    def _learn_bijective_cmap(
        self, src, dst
    ):
        if src.shape != dst.shape:
            return None
        mapping = {}
        used_dst: set[int] = set()
        for a, b in zip(src.ravel(), dst.ravel()):
            a, b = int(a), int(b)
            if a in mapping and mapping[a] != b:
                return None
            if a not in mapping:
                if b in used_dst:
                    return None
                mapping[a] = b
                used_dst.add(b)
        return mapping
    def _learn_global_bijective_cmap(
        self, training_pairs
    ):
        global_map = None
        for pair in training_pairs:
            A = pair.get_input_data().data()
            B = pair.get_output_data().data()
            color_map = self._learn_bijective_cmap(A, B)
            if color_map is None:
                return None
            if global_map is None:
                global_map = color_map
            elif global_map != color_map:
                return None
        return global_map
    def _two_step_gray_partner(self, grid: np.ndarray, gray: int = 5) -> np.ndarray:
        colors = set(map(int, np.unique(grid)))
        if gray not in colors:
            return grid.copy()
        partners = [c for c in colors if c not in {gray, 0}]
        if len(partners) != 1:
            return grid.copy()
        partner = partners[0]
        result = grid.copy()
        flat = result.ravel()
        for i, value in enumerate(flat):
            if value == gray:
                flat[i] = partner
            elif value == partner:
                flat[i] = 0
        return result
    def _expand_gray_to_3x3_blue(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        result = np.zeros((h, w), dtype=int)
        gray_positions = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 5:
                    gray_positions.append((r, c))
        for r, c in gray_positions:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr, nc] = 1
        return result
    def _move_to_matching_sides(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        if h < 3 or w < 3:
            return grid.copy()
        result = grid.copy()
        top_color = grid[0, w // 2]
        bottom_color = grid[h - 1, w // 2]
        left_color = grid[h // 2, 0]
        right_color = grid[h // 2, w - 1]
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                result[r, c] = 0
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                pixel = grid[r, c]
                if pixel == 0:
                    continue
                if pixel == top_color:
                    result[1, c] = pixel
                elif pixel == bottom_color:
                    result[h - 2, c] = pixel
                elif pixel == left_color:
                    result[r, 1] = pixel
                elif pixel == right_color:
                    result[r, w - 2] = pixel
        return result
    def _crop_and_swap_colors(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        bg = 0
        rows, cols = np.where(grid != bg)
        if len(rows) == 0:
            return grid.copy()
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        cropped = grid[min_r:max_r + 1, min_c:max_c + 1].copy()
        colors = set(cropped.flatten())
        colors.discard(bg)
        if len(colors) != 2:
            return cropped
        border_color = int(cropped[0, 0])
        center_color = None
        for c in colors:
            if c != border_color:
                center_color = int(c)
                break
        if center_color is None:
            return cropped
        result = cropped.copy()
        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                if cropped[r, c] == border_color:
                    result[r, c] = center_color
                elif cropped[r, c] == center_color:
                    result[r, c] = border_color
        return result
    def _hollow_rectangles(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        result = grid.copy()
        colors = set(grid.flatten())
        colors.discard(0)
        for color in colors:
            rows, cols = np.where(grid == color)
            if len(rows) == 0:
                continue
            visited = np.zeros((h, w), dtype=bool)
            for start_r, start_c in zip(rows, cols):
                if visited[start_r, start_c]:
                    continue
                min_r, max_r = start_r, start_r
                min_c, max_c = start_c, start_c
                stack = [(start_r, start_c)]
                visited[start_r, start_c] = True
                component_cells = [(start_r, start_c)]
                while stack:
                    r, c = stack.pop()
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                                component_cells.append((nr, nc))
                is_rectangle = True
                for r in range(min_r, max_r + 1):
                    for c in range(min_c, max_c + 1):
                        if grid[r, c] != color:
                            is_rectangle = False
                            break
                    if not is_rectangle:
                        break
                if is_rectangle:
                    for r in range(min_r + 1, max_r):
                        for c in range(min_c + 1, max_c):
                            result[r, c] = 0
        return result
    def _diagonal_expand_to_corners(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        result = np.zeros((h, w), dtype=int)
        pixel_pos = None
        pixel_color = None
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    pixel_pos = (r, c)
                    pixel_color = int(grid[r, c])
                    break
            if pixel_pos:
                break
        if pixel_pos is None:
            return grid.copy()
        start_r, start_c = pixel_pos
        r, c = start_r, start_c
        while r >= 0 and c >= 0:
            result[r, c] = pixel_color
            r -= 1
            c -= 1
        r, c = start_r, start_c
        while r >= 0 and c < w:
            result[r, c] = pixel_color
            r -= 1
            c += 1
        r, c = start_r, start_c
        while r < h and c >= 0:
            result[r, c] = pixel_color
            r += 1
            c -= 1
        r, c = start_r, start_c
        while r < h and c < w:
            result[r, c] = pixel_color
            r += 1
            c += 1
        return result
    def _gray_divider_and(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        gray_col = None
        for c in range(w):
            if all(grid[r, c] == 5 for r in range(h)):
                gray_col = c
                break
        if gray_col is None:
            return grid.copy()
        left = grid[:, :gray_col]
        right = grid[:, gray_col + 1:]
        min_h = min(left.shape[0], right.shape[0])
        min_w = min(left.shape[1], right.shape[1])
        if min_h == 0 or min_w == 0:
            return grid.copy()
        left = left[:min_h, :min_w]
        right = right[:min_h, :min_w]
        result = np.zeros((min_h, min_w), dtype=int)
        for r in range(min_h):
            for c in range(min_w):
                if left[r, c] != 0 and right[r, c] != 0:
                    result[r, c] = 2
        return result
    def _blue_divider_nor(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        blue_col = None
        for c in range(w):
            if all(grid[r, c] == 1 for r in range(h)):
                blue_col = c
                break
        if blue_col is None:
            return grid.copy()
        left = grid[:, :blue_col]
        right = grid[:, blue_col + 1:]
        min_h = min(left.shape[0], right.shape[0])
        min_w = min(left.shape[1], right.shape[1])
        if min_h == 0 or min_w == 0:
            return grid.copy()
        left = left[:min_h, :min_w]
        right = right[:min_h, :min_w]
        result = np.zeros((min_h, min_w), dtype=int)
        for r in range(min_h):
            for c in range(min_w):
                if left[r, c] == 0 and right[r, c] == 0:
                    result[r, c] = 3
        return result
    def _yellow_divider_or(self, grid):
        h, w = grid.shape
        yellow_row = None
        for r in range(h):
            if all(grid[r, c] == 4 for c in range(w)):
                yellow_row = r
                break
        if yellow_row is None:
            return grid.copy()
        top = grid[:yellow_row, :]
        bottom = grid[yellow_row + 1:, :]
        min_h = min(top.shape[0], bottom.shape[0])
        min_w = min(top.shape[1], bottom.shape[1])
        if min_h == 0 or min_w == 0:
            return grid.copy()
        top = top[:min_h, :min_w]
        bottom = bottom[:min_h, :min_w]
        result = np.zeros((min_h, min_w), dtype=int)
        for r in range(min_h):
            for c in range(min_w):
                if top[r, c] != 0 or bottom[r, c] != 0:
                    result[r, c] = 3
        return result
    def _yellow_divider_nor(self, grid):
        h, w = grid.shape
        yellow_row = None
        for r in range(h):
            if all(grid[r, c] == 4 for c in range(w)):
                yellow_row = r
                break
        if yellow_row is None:
            return grid.copy()
        top = grid[:yellow_row, :]
        bottom = grid[yellow_row + 1:, :]
        min_h = min(top.shape[0], bottom.shape[0])
        min_w = min(top.shape[1], bottom.shape[1])
        if min_h == 0 or min_w == 0:
            return grid.copy()
        top = top[:min_h, :min_w]
        bottom = bottom[:min_h, :min_w]
        result = np.zeros((min_h, min_w), dtype=int)
        for r in range(min_h):
            for c in range(min_w):
                if top[r, c] == 0 and bottom[r, c] == 0:
                    result[r, c] = 3
        return result
    def _draw_green_spiral(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        result = np.zeros((h, w), dtype=int)
        top, bottom = 0, h - 1
        left, right = 0, w - 1
        is_first_layer = True
        while top <= bottom and left <= right:
            start_col = left if is_first_layer else max(0, left - 1)
            for c in range(start_col, right + 1):
                result[top, c] = 3
            for r in range(top + 1, bottom + 1):
                result[r, right] = 3
            if top < bottom:
                for c in range(right - 1, left - 1, -1):
                    result[bottom, c] = 3
            if left < right and top + 1 < bottom:
                for r in range(bottom - 1, top + 1, -1):
                    result[r, left] = 3
            top += 2
            bottom -= 2
            left += 2
            right -= 2
            is_first_layer = False
        return result
    def _transform_candidates(self):
        return {
            "identity": lambda g: g,
            "rot90": lambda g: np.rot90(g, 1),
            "rot180": lambda g: np.rot90(g, 2),
            "rot270": lambda g: np.rot90(g, 3),
            "flip_lr": lambda g: np.fliplr(g),
            "flip_ud": lambda g: np.flipud(g),
            "transpose": lambda g: np.transpose(g),
            "mirror_lr": self._mirror_fill_left_right,
            "mirror_ud": self._mirror_fill_up_down,
            "crop_only": self._crop_bounding_box_nonbg,
            "two_step_gray_partner": self._two_step_gray_partner,
            "expand_gray_to_3x3_blue": self._expand_gray_to_3x3_blue,
            "move_to_matching_sides": self._move_to_matching_sides,
            "green_spiral": self._draw_green_spiral,
            "gray_divider_and": self._gray_divider_and,
            "blue_divider_nor": self._blue_divider_nor,
            "yellow_divider_or": self._yellow_divider_or,
            "yellow_divider_nor": self._yellow_divider_nor,
            "diagonal_expand": self._diagonal_expand_to_corners,
            "hollow_rectangles": self._hollow_rectangles,
            "crop_and_swap_colors": self._crop_and_swap_colors,
        }
    def _extra_learned_candidates(
        self, training_pairs
    ):
        extras = {}
        color_map = self._learn_global_bijective_cmap(training_pairs)
        if color_map:
            extras["recolor_bij"] = lambda grid, cmap=color_map: self._apply_color_map(
                grid, cmap
            )
        return extras
    def make_predictions(self, arc_problem):
        candidates = self._transform_candidates()
        training_pairs = list(arc_problem.training_set())
        candidates.update(self._extra_learned_candidates(training_pairs))
        scores = {name: 0 for name in candidates}
        for pair in training_pairs:
            src = pair.get_input_data().data()
            dst = pair.get_output_data().data()
            for name, transform in candidates.items():
                trial = transform(src)
                if trial.shape == dst.shape and np.array_equal(trial, dst):
                    scores[name] += 1
        if max(scores.values()) == 0:
            scores["identity"] = 1
        top3 = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        test_input = arc_problem.test_set().get_input_data().data()
        predictions = [candidates[name](test_input) for name, _ in top3]
        while len(predictions) < 3:
            predictions.append(test_input.copy())
        return predictions[:3]

def np_equal(a: np.ndarray, b: np.ndarray):
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
def bounding_box_of_foreground(grid: np.ndarray, bg: int = 0):
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
    def bbox_of_color(color):
        ys, xs = np.where(inp == color)
        if ys.size == 0:
            return None
        return ys.min(), ys.max(), xs.min(), xs.max()
    def looks_like_anchor(color, y0, y1, x0, x1):
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
def find_solid_blocks(inp: np.ndarray, block_size: int = 2):
    h, w = inp.shape
    hits = []
    for r in range(h - block_size + 1):
        for c in range(w - block_size + 1):
            patch = inp[r : r + block_size, c : c + block_size]
            vals = patch.flatten()
            if np.all(vals != 0) and np.all(vals == vals[0]):
                hits.append((int(vals[0]), r, c))
    return hits
def infer_direction_mapping(train_pairs):
    mapping = {}
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
def apply_trails(inp: np.ndarray, mapping) -> np.ndarray:
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
def rule_color_frequency_columns(inp):
    h, w = inp.shape
    bg = 0
    counts = {}
    first_pos = {}
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
def find_shapes_with_holes(inp: np.ndarray):
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
    midpoint = w // 2
    target = midpoint + (n_starting - 1)
    if target > w:
        target = w
    num_rows = target - n_starting + 1
    if num_rows <= 0:
        return inp.copy()
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
    pixels = []
    for r in range(h):
        for c in range(w):
            if inp[r, c] != bg:
                pixels.append((r, c, int(inp[r, c])))
    if len(pixels) != 4:
        return inp.copy()
    color_groups = {}
    for r, c, color in pixels:
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append((r, c))
    if len(color_groups) != 2:
        return inp.copy()
    colors = list(color_groups.keys())
    if len(color_groups[colors[0]]) != 2 or len(color_groups[colors[1]]) != 2:
        return inp.copy()
    color1, color2 = colors[0], colors[1]
    for r, c, pixel_color in pixels:
        box_color = color2 if pixel_color == color1 else color1
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if out[rr, cc] == bg:
                        out[rr, cc] = box_color
    dotted_color = 5
    for i in range(len(pixels)):
        for j in range(i + 1, len(pixels)):
            r1, c1, color1 = pixels[i]
            r2, c2, color2 = pixels[j]
            if c1 == c2:
                min_r = min(r1, r2)
                max_r = max(r1, r2)
                near_edge = min_r + 2
                far_edge = max_r - 2
                if near_edge <= far_edge and out[near_edge, c1] == bg:
                    out[near_edge, c1] = dotted_color
                if near_edge + 2 <= far_edge and out[near_edge + 2, c1] == bg:
                    out[near_edge + 2, c1] = dotted_color
                if far_edge > near_edge + 2 and out[far_edge, c1] == bg:
                    out[far_edge, c1] = dotted_color
                if far_edge - 2 > near_edge + 2 and out[far_edge - 2, c1] == bg:
                    out[far_edge - 2, c1] = dotted_color
            if r1 == r2:
                min_c = min(c1, c2)
                max_c = max(c1, c2)
                near_edge = min_c + 2
                far_edge = max_c - 2
                if near_edge <= far_edge and out[r1, near_edge] == bg:
                    out[r1, near_edge] = dotted_color
                if near_edge + 2 <= far_edge and out[r1, near_edge + 2] == bg:
                    out[r1, near_edge + 2] = dotted_color
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
BASE_RULES_C: List[Callable[[np.ndarray], np.ndarray]] = [
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
def score_rule_on_pairs(rule_fn: Callable[[np.ndarray], np.ndarray], train_pairs: List[Tuple[np.ndarray, np.ndarray]]):
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
class ArcAgentC:
    def __init__(self):
        pass
    def _get_training_pairs(self, arc_problem) -> List[Tuple[np.ndarray, np.ndarray]]:
        pairs = []
        for trn_set in arc_problem.training_set():
            tin = trn_set.get_input_data().data()
            tout = trn_set.get_output_data().data()
            pairs.append((tin, tout))
        return pairs
    def _get_test_input(self, arc_problem) -> np.ndarray:
        return arc_problem.test_set().get_input_data().data()
    def make_predictions(self, arc_problem) -> List[np.ndarray]:
        train_pairs = self._get_training_pairs(arc_problem)
        test_in = self._get_test_input(arc_problem)
        if len(train_pairs) == 0:
            return [test_in.copy()]
        dynamic_rules: List[Callable[[np.ndarray], np.ndarray]] = []
        dynamic_rules.extend(BASE_RULES_C)
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

def np_equal(a, b):
    return a.shape == b.shape and np.array_equal(a, b)
def hstack_or_pad(blocks, pad_val=0):
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
def bounding_box_of_foreground(grid, bg=0):
    ys, xs = np.where(grid != bg)
    if ys.size == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()
def extract_connected_components(grid, bg=0):
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
def rule_identity(inp):
    return inp.copy()
def rule_bounding_box_fill(inp):
    bg = 0
    bb = bounding_box_of_foreground(inp, bg=bg)
    if bb is None:
        return np.zeros_like(inp)
    y0, y1, x0, x1 = bb
    out = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=int)
    out[:, :] = inp[y0 : y1 + 1, x0 : x1 + 1]
    return out
def rule_tile2x2(inp):
    A = inp
    Ax = np.fliplr(A)
    Ay = np.flipud(A)
    Axy = np.flipud(np.fliplr(A))
    top = np.concatenate([A, Ax], axis=1)
    bot = np.concatenate([Ay, Axy], axis=1)
    return np.concatenate([top, bot], axis=0)
def rule_components_row(inp):
    comps = extract_connected_components(inp, bg=0)
    comps.sort(key=lambda g: (-np.count_nonzero(g), int(g[g != 0][0]) if np.any(g != 0) else -1))
    return hstack_or_pad(comps, pad_val=0)
def rule_merge_across_gray_line(inp):
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
def rule_transpose(inp):
    return inp.T.copy()
def rule_anchor_frame_extract(inp):
    bg = 0
    colors = [c for c in np.unique(inp) if c != bg]
    def bbox_of_color(color):
        ys, xs = np.where(inp == color)
        if ys.size == 0:
            return None
        return ys.min(), ys.max(), xs.min(), xs.max()
    def looks_like_anchor(color, y0, y1, x0, x1):
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
def find_solid_blocks(inp, block_size=2):
    h, w = inp.shape
    hits = []
    for r in range(h - block_size + 1):
        for c in range(w - block_size + 1):
            patch = inp[r : r + block_size, c : c + block_size]
            vals = patch.flatten()
            if np.all(vals != 0) and np.all(vals == vals[0]):
                hits.append((int(vals[0]), r, c))
    return hits
def infer_direction_mapping(train_pairs):
    mapping = {}
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
def apply_trails(inp, mapping):
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
def rule_diagonal_trails_from_blocks(inp, train_pairs=None):
    if train_pairs is None or len(train_pairs) == 0:
        return inp.copy()
    mapping = infer_direction_mapping(train_pairs)
    return apply_trails(inp, mapping)
def rule_fill_rows_by_matching_edges(inp):
    h, w = inp.shape
    out = inp.copy()
    for r in range(h):
        left_color = inp[r, 0]
        right_color = inp[r, w - 1]
        if left_color != 0 and left_color == right_color:
            out[r, :] = left_color
    return out
def rule_color_frequency_columns(inp):
    h, w = inp.shape
    bg = 0
    counts = {}
    first_pos = {}
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
def rule_superimpose_across_divider(inp):
    w = inp.shape[1]
    divider_col = None
    for c in range(w):
        col = inp[:, c]
        unique_vals = np.unique(col)
        if len(unique_vals) == 1 and unique_vals[0] != 0:
            divider_col = c
            break
    if divider_col is None:
        return inp.copy()
    left = inp[:, :divider_col]
    right = inp[:, divider_col + 1:]
    min_width = min(left.shape[1], right.shape[1])
    left = left[:, :min_width]
    right = right[:, :min_width]
    result = np.where(left != 0, left, right)
    result = np.where(result != 0, 1, 0)
    return result
def rule_fuse_across_gray_divider(inp):
    w = inp.shape[1]
    divider_col = None
    for c in range(w):
        col = inp[:, c]
        unique_vals = np.unique(col)
        if len(unique_vals) == 1 and unique_vals[0] == 5:
            divider_col = c
            break
    if divider_col is None:
        return inp.copy()
    left = inp[:, :divider_col]
    right = inp[:, divider_col + 1:]
    min_width = min(left.shape[1], right.shape[1])
    left = left[:, :min_width]
    right = right[:, :min_width]
    conflicts = (left != 0) & (right != 0) & (left != right)
    if np.any(conflicts):
        return left.copy()
    else:
        result = np.where(left != 0, left, right)
        return result
def find_enclosed_rings(grid: np.ndarray, bg: int = 0):
    h, w = grid.shape
    rings = []
    ring_colors = [c for c in np.unique(grid) if c != bg]
    for ring_color in ring_colors:
        visited = np.zeros((h, w), dtype=bool)
        visited[grid == ring_color] = True
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and grid[r, c] != ring_color:
                    stack = [(r, c)]
                    region = []
                    visited[r, c] = True
                    while stack:
                        rr, cc = stack.pop()
                        region.append((rr, cc))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] != ring_color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    touches_border = any(rr == 0 or rr == h-1 or cc == 0 or cc == w-1 for rr, cc in region)
                    if not touches_border and len(region) > 0:
                        rings.append((ring_color, region))
    return rings
def rule_fill_enclosed_rings(inp):
    bg = 0
    out = np.zeros_like(inp)
    rings = find_enclosed_rings(inp, bg=bg)
    ring_colors = set(ring_color for ring_color, _ in rings)
    for color in np.unique(inp):
        if color == bg:
            continue
        color_coords = set(zip(*np.where(inp == color)))
        if len(color_coords) < 8:
            continue
        rows = [r for r, c in color_coords]
        cols = [c for r, c in color_coords]
        if rows and cols:
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            top_count = sum((min_r, c) in color_coords for c in range(min_c, max_c + 1))
            bottom_count = sum((max_r, c) in color_coords for c in range(min_c, max_c + 1))
            left_count = sum((r, min_c) in color_coords for r in range(min_r, max_r + 1))
            right_count = sum((r, max_c) in color_coords for r in range(min_r, max_r + 1))
            width = max_c - min_c + 1
            height = max_r - min_r + 1
            has_top = top_count >= width * 0.4
            has_bottom = bottom_count >= width * 0.4
            has_left = left_count >= height * 0.4
            has_right = right_count >= height * 0.4
            sides_present = sum([has_top, has_bottom, has_left, has_right])
            if sides_present >= 3:
                ring_colors.add(color)
    def find_ring_boundary_and_interior_pieces(ring_color, interior_cells):
        h, w = inp.shape
        all_ring_pixels = set(zip(*np.where(inp == ring_color)))
        isolated_ring_pixels = set()
        connected_ring_pixels = set()
        for r, c in all_ring_pixels:
            has_ring_neighbor = False
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if (nr, nc) in all_ring_pixels and (nr, nc) != (r, c):
                        has_ring_neighbor = True
                        break
            if not has_ring_neighbor:
                isolated_ring_pixels.add((r, c))
            else:
                connected_ring_pixels.add((r, c))
        interior_pieces_in_this_ring = set()
        for r, c in isolated_ring_pixels:
            if (r, c) in interior_cells:
                interior_pieces_in_this_ring.add((r, c))
            else:
                surrounded_by_interior = True
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if (nr, nc) not in interior_cells and inp[nr, nc] != ring_color:
                            surrounded_by_interior = False
                            break
                if surrounded_by_interior:
                    interior_pieces_in_this_ring.add((r, c))
        return connected_ring_pixels, interior_pieces_in_this_ring
    for color in ring_colors:
        if color not in [ring_color for ring_color, _ in rings]:
            out[inp == color] = color
    ring_interior_pieces = {}
    for ring_color, region_cells in rings:
        ring_boundary, interior_pieces = find_ring_boundary_and_interior_pieces(ring_color, set(region_cells))
        for r, c in ring_boundary:
            out[r, c] = ring_color
        ring_interior_pieces[ring_color] = interior_pieces
    for ring_color, region_cells in rings:
        color_counts = {}
        for r, c in region_cells:
            val = inp[r, c]
            if val != bg and val != ring_color:
                color_counts[val] = color_counts.get(val, 0) + 1
        if not color_counts:
            for r, c in region_cells:
                out[r, c] = bg
            continue
        fill_color = max(color_counts.keys(), key=lambda c: color_counts[c])
        for r, c in region_cells:
            out[r, c] = fill_color
        if ring_color in ring_interior_pieces:
            for r, c in ring_interior_pieces[ring_color]:
                out[r, c] = fill_color
    return out
def find_plus_flowers(inp: np.ndarray, flower_color: int = 2):
    h, w = inp.shape
    flowers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if inp[r, c] == 0:
                has_up = inp[r - 1, c] == flower_color
                has_down = inp[r + 1, c] == flower_color
                has_left = inp[r, c - 1] == flower_color
                has_right = inp[r, c + 1] == flower_color
                if has_up and has_down and has_left and has_right:
                    flowers.append((r, c))
    return flowers
def rule_connect_flower_petals(inp):
    flower_color = 2
    line_color = 1
    bg = 0
    out = inp.copy()
    h, w = inp.shape
    flowers = find_plus_flowers(inp, flower_color)
    up_petals = []
    down_petals = []
    left_petals = []
    right_petals = []
    for center_r, center_c in flowers:
        up_petals.append((center_r - 1, center_c))
        down_petals.append((center_r + 1, center_c))
        left_petals.append((center_r, center_c - 1))
        right_petals.append((center_r, center_c + 1))
    for right_petal in right_petals:
        r1, c1 = right_petal
        for left_petal in left_petals:
            r2, c2 = left_petal
            if r1 == r2 and c2 > c1:
                for c in range(c1 + 1, c2):
                    if out[r1, c] == bg:
                        out[r1, c] = line_color
    for down_petal in down_petals:
        r1, c1 = down_petal
        for up_petal in up_petals:
            r2, c2 = up_petal
            if c1 == c2 and r2 > r1:
                for r in range(r1 + 1, r2):
                    if out[r, c1] == bg:
                        out[r, c1] = line_color
    return out
def find_tray_frames(inp: np.ndarray, frame_color: int = 8):
    h, w = inp.shape
    visited = np.zeros((h, w), dtype=bool)
    frames = []
    for r in range(h):
        for c in range(w):
            if inp[r, c] == frame_color and not visited[r, c]:
                stack = [(r, c)]
                component = []
                visited[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    component.append((rr, cc))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and inp[nr, nc] == frame_color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                if component:
                    rows = [p[0] for p in component]
                    cols = [p[1] for p in component]
                    r0, r1 = min(rows), max(rows)
                    c0, c1 = min(cols), max(cols)
                    if r1 - r0 >= 2 and c1 - c0 >= 2:
                        frames.append((r0, r1, c0, c1))
    return frames
def rule_reflect_shape_in_tray(inp):
    frame_color = 8
    shape_color = 2
    bg = 0
    out = inp.copy()
    frames = find_tray_frames(inp, frame_color)
    for r0, r1, c0, c1 in frames:
        interior_r0, interior_r1 = r0 + 1, r1 - 1
        interior_c0, interior_c1 = c0 + 1, c1 - 1
        if interior_r1 < interior_r0 or interior_c1 < interior_c0:
            continue
        shape_coords = []
        for r in range(interior_r0, interior_r1 + 1):
            for c in range(interior_c0, interior_c1 + 1):
                if inp[r, c] == shape_color:
                    shape_coords.append((r, c))
        if not shape_coords:
            continue
        interior_h = interior_r1 - interior_r0 + 1
        interior_w = interior_c1 - interior_c0 + 1
        mid_r = interior_r0 + interior_h // 2
        mid_c = interior_c0 + interior_w // 2
        shape_rows = [r for r, c in shape_coords]
        shape_cols = [c for r, c in shape_coords]
        avg_r = sum(shape_rows) / len(shape_rows)
        avg_c = sum(shape_cols) / len(shape_cols)
        if interior_w > interior_h:
            if avg_c < mid_c:
                for r, c in shape_coords:
                    offset_c = c - interior_c0
                    mirror_c = interior_c1 - offset_c
                    out[r, mirror_c] = shape_color
            else:
                for r, c in shape_coords:
                    offset_c = interior_c1 - c
                    mirror_c = interior_c0 + offset_c
                    out[r, mirror_c] = shape_color
        else:
            if avg_r < mid_r:
                for r, c in shape_coords:
                    offset_r = r - interior_r0
                    mirror_r = interior_r1 - offset_r
                    out[mirror_r, c] = shape_color
            else:
                for r, c in shape_coords:
                    offset_r = interior_r1 - r
                    mirror_r = interior_r0 + offset_r
                    out[mirror_r, c] = shape_color
    return out
def rule_draw_rings_around_blue_ring(inp):
    blue_color = 1
    green_color = 3
    orange_color = 2
    bg = 0
    out = inp.copy()
    h, w = inp.shape
    visited = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if inp[r, c] == blue_color and not visited[r, c]:
                stack = [(r, c)]
                component = []
                visited[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    component.append((rr, cc))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and inp[nr, nc] == blue_color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                if not component:
                    continue
                rows = [p[0] for p in component]
                cols = [p[1] for p in component]
                r0, r1 = min(rows), max(rows)
                c0, c1 = min(cols), max(cols)
                if r1 - r0 < 2 or c1 - c0 < 2:
                    continue
                component_set = set(component)
                interior_cells = set()
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        if (rr, cc) not in component_set:
                            interior_cells.add((rr, cc))
                exterior = set()
                edge_queue = []
                for rr in range(r0, r1 + 1):
                    for cc in [c0, c1]:
                        if (rr, cc) in interior_cells:
                            edge_queue.append((rr, cc))
                for cc in range(c0, c1 + 1):
                    for rr in [r0, r1]:
                        if (rr, cc) in interior_cells:
                            edge_queue.append((rr, cc))
                for cell in edge_queue:
                    if cell in interior_cells and cell not in exterior:
                        stack = [cell]
                        while stack:
                            rr, cc = stack.pop()
                            if (rr, cc) in exterior or (rr, cc) not in interior_cells:
                                continue
                            exterior.add((rr, cc))
                            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                nr, nc = rr + dr, cc + dc
                                if (nr, nc) in interior_cells and (nr, nc) not in exterior:
                                    stack.append((nr, nc))
                true_interior = interior_cells - exterior
                if not true_interior:
                    continue
                for rr, cc in component:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in true_interior:
                                if out[nr, nc] == bg:
                                    out[nr, nc] = green_color
                for rr, cc in component:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if (nr, nc) not in component_set and (nr, nc) not in true_interior:
                                    if out[nr, nc] == bg:
                                        out[nr, nc] = orange_color
    return out
def rule_count_and_sort_colored_boxes(inp):
    h, w = inp.shape
    bg = 0
    color_counts = {}
    for val in inp.flat:
        if val != bg:
            color_counts[val] = color_counts.get(val, 0) + 1
    if not color_counts:
        return inp.copy()
    grid_color = max(color_counts.items(), key=lambda x: x[1])[0]
    box_colors = []
    for r in range(h - 1):
        for c in range(w - 1):
            color = inp[r, c]
            if color == bg or color == grid_color:
                continue
            if (inp[r, c] == color and
                inp[r, c + 1] == color and
                inp[r + 1, c] == color and
                inp[r + 1, c + 1] == color):
                box_colors.append(color)
    color_box_counts = {}
    for color in box_colors:
        color_box_counts[color] = color_box_counts.get(color, 0) + 1
    sorted_colors = sorted(color_box_counts.items(), key=lambda x: (x[1], x[0]))
    if not sorted_colors:
        return np.array([[bg]])
    max_count = max(count for _, count in sorted_colors)
    num_colors = len(sorted_colors)
    out = np.zeros((num_colors, max_count), dtype=inp.dtype)
    for i, (color, count) in enumerate(sorted_colors):
        for j in range(count):
            out[i, j] = color
    return out
def rule_complete_quadrant_symmetry(inp):
    h, w = inp.shape
    bg = 0
    out = inp.copy()
    color_counts = {}
    for val in inp.flat:
        if val != bg:
            color_counts[val] = color_counts.get(val, 0) + 1
    if not color_counts:
        return out
    grid_color = max(color_counts.items(), key=lambda x: x[1])[0]
    h_lines = []
    v_lines = []
    for r in range(h):
        if all(inp[r, c] == grid_color for c in range(w)):
            h_lines.append(r)
    for c in range(w):
        if all(inp[r, c] == grid_color for r in range(h)):
            v_lines.append(c)
    if not h_lines or not v_lines:
        return out
    h_regions = []
    prev_h = 0
    for hl in h_lines:
        if hl > prev_h:
            h_regions.append((prev_h, hl))
        prev_h = hl + 1
    if prev_h < h:
        h_regions.append((prev_h, h))
    v_regions = []
    prev_v = 0
    for vl in v_lines:
        if vl > prev_v:
            v_regions.append((prev_v, vl))
        prev_v = vl + 1
    if prev_v < w:
        v_regions.append((prev_v, w))
    shape_colors = set()
    for val in inp.flat:
        if val != bg and val != grid_color:
            shape_colors.add(val)
    for shape_color in shape_colors:
        for i in range(len(h_regions) - 1):
            for j in range(len(v_regions) - 1):
                r1_start, r1_end = h_regions[i]
                r2_start, r2_end = h_regions[i + 1]
                c1_start, c1_end = v_regions[j]
                c2_start, c2_end = v_regions[j + 1]
                if (r1_end - r1_start) != (r2_end - r2_start):
                    continue
                if (c1_end - c1_start) != (c2_end - c2_start):
                    continue
                top_left = out[r1_start:r1_end, c1_start:c1_end]
                top_right = out[r1_start:r1_end, c2_start:c2_end]
                bottom_left = out[r2_start:r2_end, c1_start:c1_end]
                bottom_right = out[r2_start:r2_end, c2_start:c2_end]
                has_color_tl = np.any(top_left == shape_color)
                has_color_tr = np.any(top_right == shape_color)
                has_color_bl = np.any(bottom_left == shape_color)
                has_color_br = np.any(bottom_right == shape_color)
                color_count = sum([has_color_tl, has_color_tr, has_color_bl, has_color_br])
                if color_count == 3:
                    if not has_color_tl:
                        out[r1_start:r1_end, c1_start:c1_end] = np.flipud(np.fliplr(bottom_right))
                    elif not has_color_tr:
                        out[r1_start:r1_end, c2_start:c2_end] = np.fliplr(top_left)
                    elif not has_color_bl:
                        out[r2_start:r2_end, c1_start:c1_end] = np.flipud(top_left)
                    elif not has_color_br:
                        out[r2_start:r2_end, c2_start:c2_end] = np.flipud(np.fliplr(top_left))
    return out
def rule_fit_shapes_into_holes(inp):
    h, w = inp.shape
    bg = 0
    out = inp.copy()
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
    holes_row = None
    if base_row > 0:
        row_above = base_row - 1
        has_ground_color = np.any(inp[row_above, :] == ground_color)
        has_background = np.any(inp[row_above, :] == bg)
        if has_ground_color and has_background:
            holes_row = row_above
    if holes_row is None:
        return out
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
    out[:holes_row, :] = bg
    def get_rotations(arr):
        rots = [arr, np.rot90(arr), np.rot90(arr, 2), np.rot90(arr, 3)]
        unique_rots = []
        for rot in rots:
            is_dup = False
            for existing in unique_rots:
                if rot.shape == existing.shape and np.array_equal(rot, existing):
                    is_dup = True
                    break
            if not is_dup:
                unique_rots.append(rot)
        unique_rots.sort(key=lambda r: (r.shape[0], -r.shape[1]))
        return unique_rots
    def try_place_shapes(shape_idx, current_out, used_cols):
        if shape_idx >= len(shape_arrays):
            return current_out, True
        shape_color, shape_arr = shape_arrays[shape_idx]
        for rotation in get_rotations(shape_arr):
            sh, sw = rotation.shape
            for start_col in range(w - sw + 1):
                all_holes = True
                for sc in range(sw):
                    col = start_col + sc
                    if inp[holes_row, col] != bg or col in used_cols:
                        all_holes = False
                        break
                if not all_holes:
                    continue
                if holes_row < sh:
                    continue
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
                                if current_out[target_row, target_col] != bg:
                                    can_place = False
                                    break
                    if not can_place:
                        break
                if can_place:
                    new_out = current_out.copy()
                    new_used_cols = used_cols.copy()
                    for sr in range(sh):
                        for sc in range(sw):
                            if rotation[sr, sc] != bg:
                                target_row = holes_row - sh + 1 + sr
                                target_col = start_col + sc
                                new_out[target_row, target_col] = rotation[sr, sc]
                    for sc in range(sw):
                        if inp[holes_row, start_col + sc] == bg:
                            new_used_cols.add(start_col + sc)
                    result, success = try_place_shapes(shape_idx + 1, new_out, new_used_cols)
                    if success:
                        return result, True
        return current_out, False
    result, success = try_place_shapes(0, out, set())
    if success:
        return result
    return out
def rule_extract_box_contents(inp):
    h, w = inp.shape
    bg = 0
    box_color = 1
    box_coords = set(zip(*np.where(inp == box_color)))
    if not box_coords:
        return inp.copy()
    rows = [r for r, c in box_coords]
    cols = [c for r, c in box_coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return inp.copy()
    interior_pixels = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if inp[r, c] != bg and inp[r, c] != box_color:
                interior_pixels.append(inp[r, c])
    out = np.zeros((3, 3), dtype=inp.dtype)
    for i, pixel_val in enumerate(interior_pixels):
        if i < 9:
            row = i // 3
            col = i % 3
            out[row, col] = pixel_val
    return out
def rule_connect_blue_red_dots(inp):
    h, w = inp.shape
    bg = 0
    blue = 1
    red = 2
    green = 3
    out = inp.copy()
    blue_coords = np.where(inp == blue)
    red_coords = np.where(inp == red)
    if blue_coords[0].size == 0 or red_coords[0].size == 0:
        return out
    blue_r, blue_c = int(blue_coords[0][0]), int(blue_coords[1][0])
    red_r, red_c = int(red_coords[0][0]), int(red_coords[1][0])
    curr_r, curr_c = blue_r, blue_c
    while (curr_r, curr_c) != (red_r, red_c):
        row_dist = abs(red_r - curr_r)
        col_dist = abs(red_c - curr_c)
        dr = 1 if red_r > curr_r else -1 if red_r < curr_r else 0
        dc = 1 if red_c > curr_c else -1 if red_c < curr_c else 0
        if row_dist == 1 and col_dist > 1:
            curr_c += dc
        elif col_dist == 1 and row_dist > 1:
            curr_r += dr
        else:
            curr_r += dr
            curr_c += dc
        if (curr_r, curr_c) != (red_r, red_c) and (curr_r, curr_c) != (blue_r, blue_c):
            if 0 <= curr_r < h and 0 <= curr_c < w:
                out[curr_r, curr_c] = green
    return out
def rule_connect_red_dots_avoid_blue(inp):
    h, w = inp.shape
    bg = 0
    blue = 1
    red = 2
    green = 3
    out = inp.copy()
    red_coords = np.where(inp == red)
    if red_coords[0].size < 2:
        return out
    red1_r, red1_c = int(red_coords[0][0]), int(red_coords[1][0])
    red2_r, red2_c = int(red_coords[0][1]), int(red_coords[1][1])
    dr = 1 if red2_r > red1_r else -1 if red2_r < red1_r else 0
    dc = 1 if red2_c > red1_c else -1 if red2_c < red1_c else 0
    curr_r, curr_c = red1_r, red1_c
    while (curr_r, curr_c) != (red2_r, red2_c):
        row_dist = abs(red2_r - curr_r)
        col_dist = abs(red2_c - curr_c)
        step_dr = 1 if red2_r > curr_r else -1 if red2_r < curr_r else 0
        step_dc = 1 if red2_c > curr_c else -1 if red2_c < curr_c else 0
        if row_dist > 0 and col_dist > 0:
            curr_r += step_dr
            curr_c += step_dc
        elif row_dist > 0:
            curr_r += step_dr
        elif col_dist > 0:
            curr_c += step_dc
        else:
            break
        if (curr_r, curr_c) == (red2_r, red2_c):
            break
        if 0 <= curr_r < h and 0 <= curr_c < w:
            if out[curr_r, curr_c] == blue:
                out[curr_r, curr_c] = green
            elif out[curr_r, curr_c] == bg:
                out[curr_r, curr_c] = red
    return out
def rule_expanding_triangle_with_diagonals(inp):
    h, w = inp.shape
    if h != 1:
        return inp.copy()
    red_coords = np.where(inp[0] == 2)
    if red_coords[0].size == 0:
        return inp.copy()
    red_col = int(red_coords[0][0])
    left_dist = red_col
    right_dist = w - 1 - red_col
    max_dist = max(left_dist, right_dist)
    total_height = 2 * max_dist + 1
    out = np.zeros((total_height, w), dtype=inp.dtype)
    for row in range(max_dist + 1):
        if row == 0:
            out[row, red_col] = 2
        else:
            left_edge = red_col - row
            right_edge = red_col + row
            if 0 <= left_edge < w:
                out[row, left_edge] = 2
            if 0 <= right_edge < w and right_edge != left_edge:
                out[row, right_edge] = 2
    start_row = 3
    start_col = red_col - 1
    diag_num = 0
    while True:
        curr_start_row = start_row + 2 * diag_num
        curr_start_col = start_col - 2 * diag_num
        if curr_start_col < 0:
            row_offset = -curr_start_col
            curr_start_col = 0
            curr_start_row = curr_start_row + row_offset
        if curr_start_row >= total_height:
            break
        r = curr_start_row
        c = curr_start_col
        while r < total_height and c < w:
            if out[r, c] == 0:
                out[r, c] = 1
            r += 1
            c += 1
        diag_num += 1
    return out
def rule_rotate_and_tile_3x3(inp):
    h, w = inp.shape
    rotated = np.rot90(inp, 2)
    out = np.zeros((h * 3, w * 3), dtype=inp.dtype)
    for tile_row in range(3):
        for tile_col in range(3):
            tile = rotated.copy()
            if tile_col == 1:
                tile = np.fliplr(tile)
            if tile_row == 1:
                tile = np.flipud(tile)
            out[tile_row * h:(tile_row + 1) * h,
                tile_col * w:(tile_col + 1) * w] = tile
    return out
def rule_color_swap_with_legend(inp):
    h, w = inp.shape
    unique, counts = np.unique(inp[inp != 0], return_counts=True)
    if len(unique) == 0:
        return inp.copy()
    sorted_indices = np.argsort(-counts)
    sorted_colors = unique[sorted_indices]
    frame_color = None
    frame_bounds = None
    for candidate in sorted_colors:
        coords = np.argwhere(inp == candidate)
        if len(coords) == 0:
            continue
        min_r, min_c = coords.min(axis=0)
        max_r, max_c = coords.max(axis=0)
        if (max_r - min_r) >= 3 and (max_c - min_c) >= 3:
            frame_color = candidate
            frame_bounds = (min_r, min_c, max_r, max_c)
            break
    if frame_color is None:
        return inp.copy()
    min_r, min_c, max_r, max_c = frame_bounds
    region = inp[min_r:max_r+1, min_c:max_c+1].copy()
    color_map = {}
    for r in range(h):
        for c in range(w):
            if min_r <= r <= max_r and min_c <= c <= max_c:
                continue
            if inp[r, c] != 0 and inp[r, c] != frame_color:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not (min_r <= nr <= max_r and min_c <= nc <= max_c):
                            if inp[nr, nc] != 0 and inp[nr, nc] != frame_color and inp[nr, nc] != inp[r, c]:
                                color_map[inp[r, c]] = inp[nr, nc]
    for old_color, new_color in color_map.items():
        region[region == old_color] = new_color
    return region
def rule_xor_overlay_halves(inp):
    h, w = inp.shape
    if h % 2 != 0:
        return inp.copy()
    mid = h // 2
    top = inp[:mid, :]
    bottom = inp[mid:, :]
    out = np.zeros_like(top)
    top_nonzero = (top != 0)
    bottom_nonzero = (bottom != 0)
    xor_mask = top_nonzero ^ bottom_nonzero
    out[xor_mask] = 6
    return out
class DiagonalTrailRule:
    def __init__(self, train_pairs):
        self.train_pairs = train_pairs
    def __call__(self, inp):
        return rule_diagonal_trails_from_blocks(inp, self.train_pairs)
BASE_RULES_D = [
    rule_identity,
    rule_transpose,
    rule_bounding_box_fill,
    rule_tile2x2,
    rule_components_row,
    rule_merge_across_gray_line,
    rule_anchor_frame_extract,
    rule_fill_rows_by_matching_edges,
    rule_color_frequency_columns,
    rule_superimpose_across_divider,
    rule_fuse_across_gray_divider,
    rule_fill_enclosed_rings,
    rule_connect_flower_petals,
    rule_reflect_shape_in_tray,
    rule_draw_rings_around_blue_ring,
    rule_count_and_sort_colored_boxes,
    rule_complete_quadrant_symmetry,
    rule_fit_shapes_into_holes,
    rule_extract_box_contents,
    rule_connect_blue_red_dots,
    rule_connect_red_dots_avoid_blue,
    rule_expanding_triangle_with_diagonals,
    rule_rotate_and_tile_3x3,
    rule_color_swap_with_legend,
    rule_xor_overlay_halves,
]
def score_rule_on_pairs(rule_fn, train_pairs):
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
class ArcAgentD:
    def __init__(self):
        pass
    def _get_training_pairs(self, arc_problem):
        pairs = []
        for trn_set in arc_problem.training_set():
            tin = trn_set.get_input_data().data()
            tout = trn_set.get_output_data().data()
            pairs.append((tin, tout))
        return pairs
    def _get_test_input(self, arc_problem):
        return arc_problem.test_set().get_input_data().data()
    def make_predictions(self, arc_problem):
        train_pairs = self._get_training_pairs(arc_problem)
        test_in = self._get_test_input(arc_problem)
        if len(train_pairs) == 0:
            return [test_in.copy()]
        dynamic_rules = []
        dynamic_rules.extend(BASE_RULES_D)
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

class ArcAgent:
    def __init__(self):
        self.agent_b = ArcAgentB()
        self.agent_c = ArcAgentC()
        self.agent_d = ArcAgentD()

    def make_predictions(self, arc_problem):
        all_preds = []
        seen = set()
        for agent in [self.agent_b, self.agent_c, self.agent_d]:
            try:
                preds = agent.make_predictions(arc_problem)
                if preds:
                    pred = preds[0]
                    sig = (pred.shape, pred.tobytes())
                    if sig not in seen:
                        all_preds.append(pred)
                        seen.add(sig)
            except:
                pass
        if len(all_preds) < 3:
            for agent in [self.agent_b, self.agent_c, self.agent_d]:
                try:
                    preds = agent.make_predictions(arc_problem)
                    for pred in preds[1:]:
                        sig = (pred.shape, pred.tobytes())
                        if sig not in seen:
                            all_preds.append(pred)
                            seen.add(sig)
                            if len(all_preds) >= 3:
                                return all_preds[:3]
                except:
                    pass
        if len(all_preds) < 3:
            test_input = arc_problem.test_set().get_input_data().data()
            while len(all_preds) < 3:
                all_preds.append(test_input.copy())
        return all_preds[:3]
