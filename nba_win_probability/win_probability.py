import logging
import json
import pandas as pd

from collections import namedtuple
from simulation_methods.models import BrownianMotion
from typing import Tuple

logger = logging.getLogger(__name__)


class WinProbabilitySimulation(object):

    WinProbabilityOutcome = namedtuple("SimulatedResult", ["estimated_win_probability", "replications"])

    def estimate_home_win_probability(self) -> WinProbabilityOutcome:
        raise NotImplementedError


class BrownianWinProbabilitySimulation(WinProbabilitySimulation):

    def __init__(self, scores_by_minute: pd.Series, seed=None):
        self.mean, self.stdev = self._estimate_model_parameters(scores_by_minute)
        self.brownian_motion = BrownianMotion(self.mean, self.stdev, seed)

    @staticmethod
    def _estimate_model_parameters(scores_by_minute: pd.Series) -> Tuple:
        return scores_by_minute.mean(), scores_by_minute.std()

    @classmethod
    def load_serialized_model(cls, filename='../trained_models/trained_brownian_win_probability.json'):
        self = cls.__new__(cls)
        with open(filename, 'r') as f:
            model_params = json.load(f)

        self.mean = model_params['mean']
        self.stdev = model_params['stdev']
        self.brownian_motion = BrownianMotion(self.mean, self.stdev)
        return self

    def serialize(self, filename='../trained_models/trained_brownian_win_probability.json'):
        model_params = {'mean': self.mean, 'stdev': self.stdev}
        with open(filename, 'w') as f:
            json.dump(model_params, f)

    def estimate_home_win_probability(self, score_difference: int = 0, time_remaining: int = 48,
                                      num_simulations: int = 1000) -> WinProbabilitySimulation.WinProbabilityOutcome:
        replications = []
        results = []
        for i in range(num_simulations):
            brownian_outcome = self.brownian_motion.run_simulation(
                num_steps=time_remaining,
                initial_value=score_difference
            )

            replications.append(brownian_outcome.history_vector)
            results.append(self._evaluate_home_team_result(brownian_outcome.result))

        estimated_win_probability = sum(results) / len(replications)
        return WinProbabilitySimulation.WinProbabilityOutcome(
            estimated_win_probability=estimated_win_probability,
            replications=replications
        )

    @staticmethod
    def _evaluate_home_team_result(score_margin: float):
        if score_margin > 0:
            return 1
        else:
            return 0
