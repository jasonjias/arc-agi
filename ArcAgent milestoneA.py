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

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        # 1. tranform function and score dictionary
        transform_functions = {
            "rot0":       lambda x: x,
            "rot90":      lambda x: np.rot90(x, 1),
            "rot180":     lambda x: np.rot90(x, 2),
            "rot270":     lambda x: np.rot90(x, 3),
            "flip_lr":    lambda x: np.fliplr(x),
            "flip_ud":    lambda x: np.flipud(x),
            "transpose":  lambda x: np.transpose(x),
            "t_rot90":    lambda x: np.rot90(np.transpose(x), 1),
        }

        scores = {k: 0 for k in transform_functions.keys()}

        # 2) score each transform on the training pairs
        for pair in arc_problem.training_set():
            A = pair.get_input_data().data()
            B = pair.get_output_data().data()   # BUG FIX: compare to OUTPUT

            for name, fn in transform_functions.items():
                GA = fn(A)
                if GA.shape == B.shape and np.array_equal(GA, B):
                    scores[name] += 1

        # 3) take top-3 most consistent transforms
        top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]

        # 4) apply top-3 to test input
        testA = arc_problem.test_set().get_input_data().data()
        predictions = []
        for name, _ in top3:
            predictions.append(transform_functions[name](testA))

        # pad with identity if fewer than 3 (very rare)
        while len(predictions) < 3:
            predictions.append(testA.copy())

        return predictions[:3]
