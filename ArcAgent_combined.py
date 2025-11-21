import numpy as np
import sys
sys.path.insert(0, '.')

from ArcProblem import ArcProblem
from ArcAgent_milestoneB import ArcAgent as ArcAgentB
from ArcAgent_milestoneC import ArcAgent as ArcAgentC
from ArcAgent_milestoneD import ArcAgent as ArcAgentD


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
