import numpy as np
from typing import Callable, Dict, List, Optional

from ArcProblem import ArcProblem
from ArcSet import ArcSet


class ArcAgent:
    def __init__(self) -> None:
        pass

    def _mode_color(self, grid: np.ndarray) -> int:
        # Return the most frequent color ID (0-9) in grid
        values, counts = np.unique(grid, return_counts=True)
        return int(values[np.argmax(counts)])

    def _crop_bounding_box_nonbg(self, grid: np.ndarray) -> np.ndarray:
        # Crop the grid to a smaller bounding box containing all colored pixels.
        background = self._mode_color(grid)
        ys, xs = np.where(grid != background)
        if ys.size == 0:
            return grid.copy()

        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        return grid[y0:y1, x0:x1]

    def _mirror_fill_left_right(self, grid: np.ndarray) -> np.ndarray:
        # Check if reflected horizontally (left or right)
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

    def _mirror_fill_up_down(self, grid: np.ndarray) -> np.ndarray:
        # Check if reflected vertically (up or down)
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
        self, grid: np.ndarray, color_map: Dict[int, int]
    ) -> np.ndarray:
        # Apply a color map to the grid in one pass.
        result = grid.copy()
        flat = result.ravel()
        for i, value in enumerate(flat):
            flat[i] = color_map.get(int(value), int(value))
        return result

    def _learn_bijective_cmap(
        self, src: np.ndarray, dst: np.ndarray
    ) -> Optional[Dict[int, int]]:
        if src.shape != dst.shape:
            return None

        mapping: Dict[int, int] = {}
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
        self, training_pairs: List[ArcSet]
    ) -> Optional[Dict[int, int]]:
        # Check if 1 color change aligns with results
        global_map: Optional[Dict[int, int]] = None

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
        # Changes 2 color grids; gray (5) to partner color and partner color to black (0)

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
        # Find all 1x1 gray (5) pixels and replace each with a 3x3 blue (1) block
        h, w = grid.shape
        result = np.zeros((h, w), dtype=int)

        # Find all gray pixels
        gray_positions = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 5:
                    gray_positions.append((r, c))

        # Replace each gray pixel with a 3x3 blue block
        for r, c in gray_positions:
            # Draw 3x3 block centered at (r, c)
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr, nc] = 1

        return result

    def _move_to_matching_sides(self, grid: np.ndarray) -> np.ndarray:
        # Move colored pixels from interior to their matching border sides
        h, w = grid.shape
        if h < 3 or w < 3:
            return grid.copy()

        result = grid.copy()

        # Identify border colors (excluding corners)
        top_color = grid[0, w // 2]  # Sample from middle of top border
        bottom_color = grid[h - 1, w // 2]  # Sample from middle of bottom border
        left_color = grid[h // 2, 0]  # Sample from middle of left border
        right_color = grid[h // 2, w - 1]  # Sample from middle of right border

        # Clear interior (keep borders)
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                result[r, c] = 0

        # Process each interior pixel
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                pixel = grid[r, c]
                if pixel == 0:
                    continue

                # Move pixel to matching border side
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
        # Find the colored square, crop it, and swap border/center colors
        h, w = grid.shape
        bg = 0

        # Find bounding box of non-background pixels
        rows, cols = np.where(grid != bg)
        if len(rows) == 0:
            return grid.copy()

        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()

        # Crop to the bounding box
        cropped = grid[min_r:max_r + 1, min_c:max_c + 1].copy()

        # Identify the two colors: border and center
        colors = set(cropped.flatten())
        colors.discard(bg)

        if len(colors) != 2:
            return cropped

        # The border color is the one at the edges
        border_color = int(cropped[0, 0])
        center_color = None
        for c in colors:
            if c != border_color:
                center_color = int(c)
                break

        if center_color is None:
            return cropped

        # Swap colors: border -> center_color, center -> border_color
        result = cropped.copy()
        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                if cropped[r, c] == border_color:
                    result[r, c] = center_color
                elif cropped[r, c] == center_color:
                    result[r, c] = border_color

        return result

    def _hollow_rectangles(self, grid: np.ndarray) -> np.ndarray:
        # Find all rectangles and hollow them out, keeping only 1-pixel borders
        h, w = grid.shape
        result = grid.copy()

        # For each non-background color, find its rectangular regions and hollow them
        colors = set(grid.flatten())
        colors.discard(0)  # Remove background

        for color in colors:
            # Find bounding boxes for this color
            rows, cols = np.where(grid == color)
            if len(rows) == 0:
                continue

            # Group connected regions of this color
            visited = np.zeros((h, w), dtype=bool)

            for start_r, start_c in zip(rows, cols):
                if visited[start_r, start_c]:
                    continue

                # Find bounding box of this connected component
                min_r, max_r = start_r, start_r
                min_c, max_c = start_c, start_c

                # Use flood fill to find the extent of this rectangle
                stack = [(start_r, start_c)]
                visited[start_r, start_c] = True
                component_cells = [(start_r, start_c)]

                while stack:
                    r, c = stack.pop()
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

                    # Check 4-connected neighbors
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                                component_cells.append((nr, nc))

                # Check if this is a filled rectangle (all cells in bounding box are this color)
                is_rectangle = True
                for r in range(min_r, max_r + 1):
                    for c in range(min_c, max_c + 1):
                        if grid[r, c] != color:
                            is_rectangle = False
                            break
                    if not is_rectangle:
                        break

                if is_rectangle:
                    # Hollow out the interior, keep only the border
                    for r in range(min_r + 1, max_r):
                        for c in range(min_c + 1, max_c):
                            result[r, c] = 0

        return result

    def _diagonal_expand_to_corners(self, grid: np.ndarray) -> np.ndarray:
        # Find the single colored pixel and expand diagonally to all 4 corners
        h, w = grid.shape
        result = np.zeros((h, w), dtype=int)

        # Find the single colored pixel
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

        # Draw diagonal lines to all 4 corners
        # Top-left diagonal (up-left)
        r, c = start_r, start_c
        while r >= 0 and c >= 0:
            result[r, c] = pixel_color
            r -= 1
            c -= 1

        # Top-right diagonal (up-right)
        r, c = start_r, start_c
        while r >= 0 and c < w:
            result[r, c] = pixel_color
            r -= 1
            c += 1

        # Bottom-left diagonal (down-left)
        r, c = start_r, start_c
        while r < h and c >= 0:
            result[r, c] = pixel_color
            r += 1
            c -= 1

        # Bottom-right diagonal (down-right)
        r, c = start_r, start_c
        while r < h and c < w:
            result[r, c] = pixel_color
            r += 1
            c += 1

        return result

    def _gray_divider_and(self, grid: np.ndarray) -> np.ndarray:
        # Find gray (5) divider column, split left/right, AND them
        h, w = grid.shape

        # Find the gray divider column
        gray_col = None
        for c in range(w):
            if all(grid[r, c] == 5 for r in range(h)):
                gray_col = c
                break

        if gray_col is None:
            return grid.copy()

        # Split into left and right sides
        left = grid[:, :gray_col]
        right = grid[:, gray_col + 1:]

        # Ensure they have the same shape
        if left.shape != right.shape:
            return grid.copy()

        # Perform AND operation: output 2 where both sides have non-zero
        result = np.zeros(left.shape, dtype=int)
        for r in range(left.shape[0]):
            for c in range(left.shape[1]):
                if left[r, c] != 0 and right[r, c] != 0:
                    result[r, c] = 2

        return result

    def _blue_divider_nor(self, grid: np.ndarray) -> np.ndarray:
        # Find blue (1) divider column, split left/right, mark uncovered areas as green
        h, w = grid.shape

        # Find the blue divider column
        blue_col = None
        for c in range(w):
            if all(grid[r, c] == 1 for r in range(h)):
                blue_col = c
                break

        if blue_col is None:
            return grid.copy()

        # Split into left and right sides
        left = grid[:, :blue_col]
        right = grid[:, blue_col + 1:]

        # Ensure they have the same shape
        if left.shape != right.shape:
            return grid.copy()

        # Perform NOR operation: output 3 (green) where both sides are zero
        result = np.zeros(left.shape, dtype=int)
        for r in range(left.shape[0]):
            for c in range(left.shape[1]):
                if left[r, c] == 0 and right[r, c] == 0:
                    result[r, c] = 3

        return result

    def _yellow_divider_or(self, grid: np.ndarray) -> np.ndarray:
        # Find yellow (4) divider row, split top/bottom, OR them
        h, w = grid.shape

        # Find the yellow divider row
        yellow_row = None
        for r in range(h):
            if all(grid[r, c] == 4 for c in range(w)):
                yellow_row = r
                break

        if yellow_row is None:
            return grid.copy()

        # Split into top and bottom sections
        top = grid[:yellow_row, :]
        bottom = grid[yellow_row + 1:, :]

        # Ensure they have the same shape
        if top.shape != bottom.shape:
            return grid.copy()

        # Perform OR operation: output 3 (green) where either side has non-zero
        result = np.zeros(top.shape, dtype=int)
        for r in range(top.shape[0]):
            for c in range(top.shape[1]):
                if top[r, c] != 0 or bottom[r, c] != 0:
                    result[r, c] = 3

        return result

    def _yellow_divider_nor(self, grid: np.ndarray) -> np.ndarray:
        # Find yellow (4) divider row, split top/bottom, mark uncovered areas as green
        h, w = grid.shape

        # Find the yellow divider row
        yellow_row = None
        for r in range(h):
            if all(grid[r, c] == 4 for c in range(w)):
                yellow_row = r
                break

        if yellow_row is None:
            return grid.copy()

        # Split into top and bottom sections
        top = grid[:yellow_row, :]
        bottom = grid[yellow_row + 1:, :]

        # Ensure they have the same shape
        if top.shape != bottom.shape:
            return grid.copy()

        # Perform NOR operation: output 3 (green) where both sides are zero
        result = np.zeros(top.shape, dtype=int)
        for r in range(top.shape[0]):
            for c in range(top.shape[1]):
                if top[r, c] == 0 and bottom[r, c] == 0:
                    result[r, c] = 3

        return result

    def _draw_green_spiral(self, grid: np.ndarray) -> np.ndarray:
        # Draw a green (3) spiral going clockwise inward from outside
        # Skip every other layer by incrementing boundaries by 2
        h, w = grid.shape
        result = np.zeros((h, w), dtype=int)

        # Define boundaries
        top, bottom = 0, h - 1
        left, right = 0, w - 1
        is_first_layer = True

        while top <= bottom and left <= right:
            # Draw top row (left to right)
            # For non-first layers, extend left by 1 to continue the spiral
            start_col = left if is_first_layer else max(0, left - 1)
            for c in range(start_col, right + 1):
                result[top, c] = 3

            # Draw right column (top to bottom, excluding top corner)
            for r in range(top + 1, bottom + 1):
                result[r, right] = 3

            # Draw bottom row (right to left, excluding right corner)
            if top < bottom:
                for c in range(right - 1, left - 1, -1):
                    result[bottom, c] = 3

            # Draw left column (bottom to top, excluding both corners and stopping before next drawn layer)
            if left < right and top + 1 < bottom:
                for r in range(bottom - 1, top + 1, -1):
                    result[r, left] = 3

            # Move inward by 2 to skip the next layer
            top += 2
            bottom -= 2
            left += 2
            right -= 2
            is_first_layer = False

        return result

    # ─────────────────────────────────── candidate builders ─────────────────────────────────── #
    def _transform_candidates(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
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
        self, training_pairs: List[ArcSet]
    ) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        extras: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

        color_map = self._learn_global_bijective_cmap(training_pairs)
        if color_map:
            extras["recolor_bij"] = lambda grid, cmap=color_map: self._apply_color_map(
                grid, cmap
            )

        return extras

    # ────────────────────────────────────────── main function ────────────────────────────────────────── #
    def make_predictions(self, arc_problem: ArcProblem) -> List[np.ndarray]:
        """
        Voting pipeline:
          1) Build a dictionary of candidate single-step transforms.
          2) Score each candidate across all training pairs: +1 for an exact match (shape & values).
          3) Keep the top-3 scoring candidates.
          4) Apply those three to the test input and return the three predictions.
        """
        # 1) Build candidates (always-available + any learned from training)
        candidates = self._transform_candidates()
        training_pairs = list(arc_problem.training_set())
        candidates.update(self._extra_learned_candidates(training_pairs))

        # 2) Score each candidate against the training pairs
        scores: Dict[str, int] = {name: 0 for name in candidates}
        for pair in training_pairs:
            src = pair.get_input_data().data()
            dst = pair.get_output_data().data()

            for name, transform in candidates.items():
                trial = transform(src)
                if trial.shape == dst.shape and np.array_equal(trial, dst):
                    scores[name] += 1

        # Safety: if nobody scored, fall back to identity so we still produce outputs
        if max(scores.values()) == 0:
            scores["identity"] = 1

        # 3) Pick the top-3 candidates (stable tie-break by name)
        top3 = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:3]

        # 4) Apply them to the test input
        test_input = arc_problem.test_set().get_input_data().data()
        predictions = [candidates[name](test_input) for name, _ in top3]

        # Pad with copies of the input if fewer than 3
        while len(predictions) < 3:
            predictions.append(test_input.copy())

        return predictions[:3]
