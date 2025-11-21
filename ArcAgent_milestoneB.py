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
