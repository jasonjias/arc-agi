import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init. Be aware that it gets called only once
        and then the solve method will get called several times.
        """
        pass

    # ---------- basic geometries ----------
    def _geoms(self):
        return {
            "rot0":      lambda x: x,
            "rot90":     lambda x: np.rot90(x, 1),
            "rot180":    lambda x: np.rot90(x, 2),
            "rot270":    lambda x: np.rot90(x, 3),
            "flip_lr":   lambda x: np.fliplr(x),
            "flip_ud":   lambda x: np.flipud(x),
            "transpose": lambda x: np.transpose(x),
        }

    # ---------- helpers ----------
    def _mode_color(self, X: np.ndarray) -> int:
        vals, counts = np.unique(X, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _bbox_nonbg(self, X: np.ndarray, bg: int) -> tuple | None:
        ys, xs = np.where(X != bg)
        if ys.size == 0:
            return None
        return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

    def _crop_bbox_nonbg(self, X: np.ndarray):
        bg = self._mode_color(X)
        box = self._bbox_nonbg(X, bg)
        if box is None:
            return X.copy()
        y0, y1, x0, x1 = box
        return X[y0:y1, x0:x1]

    # bijective recolor (1→1)
    def _learn_cmap_bij(self, A: np.ndarray, B: np.ndarray):
        if A.shape != B.shape:
            return None
        cmap, used = {}, set()
        for a, b in zip(A.ravel(), B.ravel()):
            if a in cmap and cmap[a] != b:
                return None
            if a not in cmap:
                if b in used:
                    return None
                cmap[a] = b
                used.add(b)
        return cmap

    # surjective recolor (many→one allowed, but still deterministic)
    def _learn_cmap_surj(self, A: np.ndarray, B: np.ndarray):
        if A.shape != B.shape:
            return None
        cmap = {}
        for a, b in zip(A.ravel(), B.ravel()):
            if a in cmap and cmap[a] != b:
                return None
            cmap.setdefault(a, b)
        return cmap

    def _apply_cmap(self, X: np.ndarray, cmap: dict):
        out = X.copy()
        flat = out.ravel()
        for i, v in enumerate(flat):
            flat[i] = cmap.get(v, v)
        return out

    # ---------- special-case transforms ----------
    def _mirror_fill_lr(self, X: np.ndarray) -> np.ndarray:
        """Fill the empty half horizontally by mirroring the filled half."""
        H, W = X.shape
        out = X.copy()
        mid = W // 2
        left = out[:, :mid]
        right = out[:, W - mid:]

        # detect which side is mostly background
        bg = self._mode_color(out)
        left_nonbg = np.count_nonzero(left != bg)
        right_nonbg = np.count_nonzero(right != bg)

        if right_nonbg == 0 and left_nonbg > 0:
            out[:, W - mid:] = np.fliplr(out[:, :mid])
        elif left_nonbg == 0 and right_nonbg > 0:
            out[:, :mid] = np.fliplr(out[:, W - mid:])
        else:
            # if both sides have content, try enforcing bilateral symmetry
            # by copying the side with more content to the other.
            if left_nonbg >= right_nonbg:
                out[:, W - mid:] = np.fliplr(out[:, :mid])
            else:
                out[:, :mid] = np.fliplr(out[:, W - mid:])
        return out

    def _mirror_fill_ud(self, X: np.ndarray) -> np.ndarray:
        """Fill the empty half vertically by mirroring the filled half."""
        H, W = X.shape
        out = X.copy()
        mid = H // 2
        top = out[:mid, :]
        bot = out[H - mid:, :]

        bg = self._mode_color(out)
        top_nonbg = np.count_nonzero(top != bg)
        bot_nonbg = np.count_nonzero(bot != bg)

        if bot_nonbg == 0 and top_nonbg > 0:
            out[H - mid:, :] = np.flipud(out[:mid, :])
        elif top_nonbg == 0 and bot_nonbg > 0:
            out[:mid, :] = np.flipud(out[H - mid:, :])
        else:
            if top_nonbg >= bot_nonbg:
                out[H - mid:, :] = np.flipud(out[:mid, :])
            else:
                out[:mid, :] = np.flipud(out[H - mid:, :])
        return out

    def _learn_two_step_recolor(self, A: np.ndarray, B: np.ndarray):
        """
        Find colors (x, y) so that applying the cmap {x->y, y->0} to A yields B.
        Tries all ARC colors 0..9 (with y!=0 and x!=y). Returns (x, y) or None.
        """
        if A.shape != B.shape:
            return None

        for x in range(10):
            for y in range(10):
                if y == 0 or x == y:
                    continue
                cand = self._apply_cmap(A, {x: y, y: 0})
                if np.array_equal(cand, B):
                    return (x, y)
        return None

    def _apply_two_step_recolor(self, X: np.ndarray, pair_xy: tuple[int, int]) -> np.ndarray:
        x, y = pair_xy
        # do it via a single cmap to ensure the mapping is simultaneous
        return self._apply_cmap(X, {x: y, y: 0})

    # ---------- hypothesis application ----------
    def _apply_hypothesis(self, X, hyp):
        """
        hyp is a tuple among:
          - (rule, geom_name, cmap_items_or_None)
          - ("mirror_fill_lr",)
          - ("mirror_fill_ud",)
          - ("two_step_recolor", x, y)
        """
        rule = hyp[0]

        if rule in {"geom_only", "geom_bij", "geom_surj", "bbox_bij", "bbox_surj"}:
            _, gname, cmap_items = hyp
            G = self._geoms()[gname](X)
            if rule == "geom_only":
                return G
            elif rule in {"geom_bij", "geom_surj"}:
                return self._apply_cmap(G, dict(cmap_items))
            elif rule in {"bbox_bij", "bbox_surj"}:
                Gc = self._crop_bbox_nonbg(G)
                return self._apply_cmap(Gc, dict(cmap_items)) if cmap_items is not None else Gc

        elif rule == "mirror_fill_lr":
            return self._mirror_fill_lr(X)
        elif rule == "mirror_fill_ud":
            return self._mirror_fill_ud(X)
        elif rule == "two_step_recolor":
            _, x, y = hyp
            return self._apply_two_step_recolor(X, (x, y))

        # fallback
        return X

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        print(f"\n\nproblem id: {arc_problem.problem_name()}")
        print(f"\n\ninital problem 1: {arc_problem._training_data[0].get_input_data()}")
        print(f"\n\ninital problem 2: {arc_problem._training_data[1].get_input_data()}")
        geoms = self._geoms()
        scores = {}

        # --------- TRAIN: vote consistent rules ----------
        for pair in arc_problem.training_set():
            A = pair.get_input_data().data()
            B = pair.get_output_data().data()

            # --- Try special-case rules first ---

            # Mirror-fill Left/Right
            if A.shape == B.shape and np.array_equal(self._mirror_fill_lr(A), B):
                key = ("mirror_fill_lr",)
                scores[key] = scores.get(key, 0) + 1

            # Mirror-fill Up/Down
            if A.shape == B.shape and np.array_equal(self._mirror_fill_ud(A), B):
                key = ("mirror_fill_ud",)
                scores[key] = scores.get(key, 0) + 1

            # Two-step recolor (x->y, y->0, others identity)
            ts_pair = self._learn_two_step_recolor(A, B)
            if ts_pair is not None:
                key = ("two_step_recolor", ts_pair[0], ts_pair[1])
                scores[key] = scores.get(key, 0) + 1

            # --- Generic geometry/recolor rules ---

            for gname, gfn in geoms.items():
                GA = gfn(A)

                # (1) geometry only
                if GA.shape == B.shape and np.array_equal(GA, B):
                    key = ("geom_only", gname, None)
                    scores[key] = scores.get(key, 0) + 1

                # (2) geometry + recolor
                if GA.shape == B.shape:
                    cmap_bij = self._learn_cmap_bij(GA, B)
                    if cmap_bij is not None:
                        key = ("geom_bij", gname, frozenset(cmap_bij.items()))
                        scores[key] = scores.get(key, 0) + 1
                    else:
                        cmap_surj = self._learn_cmap_surj(GA, B)
                        if cmap_surj is not None:
                            key = ("geom_surj", gname, frozenset(cmap_surj.items()))
                            scores[key] = scores.get(key, 0) + 1

                # (3) crop bbox of non-background after geometry + recolor
                GAb = self._crop_bbox_nonbg(GA)
                if GAb.shape == B.shape:
                    # crop-only
                    if np.array_equal(GAb, B):
                        key = ("bbox_bij", gname, None)
                        scores[key] = scores.get(key, 0) + 1
                    # crop + recolor (bijective or surjective)
                    cmap_bij_b = self._learn_cmap_bij(GAb, B)
                    if cmap_bij_b is not None:
                        key = ("bbox_bij", gname, frozenset(cmap_bij_b.items()))
                        scores[key] = scores.get(key, 0) + 1
                    else:
                        cmap_surj_b = self._learn_cmap_surj(GAb, B)
                        if cmap_surj_b is not None:
                            key = ("bbox_surj", gname, frozenset(cmap_surj_b.items()))
                            scores[key] = scores.get(key, 0) + 1

        if not scores:
            scores[("geom_only", "rot0", None)] = 1

        # --------- pick top-3 ----------
        top3 = [h for (h, _) in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]]

        # --------- TEST: apply ----------
        testA = arc_problem.test_set().get_input_data().data()
        preds = []
        for hyp in top3:
            preds.append(self._apply_hypothesis(testA, hyp))

        while len(preds) < 3:
            preds.append(testA.copy())

        print(f"Predictions: {preds[:3]}")
        return preds[:3]