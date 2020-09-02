import pytest

import pandas as pd

from nba_win_probability.win_probability import WinProbabilitySimulation, BrownianWinProbabilitySimulation


@pytest.fixture
def scores_by_minute():
    return pd.Series([1, 2, 3])


@pytest.fixture
def brownian_wp(scores_by_minute):
    return BrownianWinProbabilitySimulation(scores_by_minute)


def test_estimate_model_parameters(scores_by_minute):
    # scores_by_minute =
    mean, stdev = BrownianWinProbabilitySimulation._estimate_model_parameters(scores_by_minute)

    assert mean == 2.0
    assert stdev == 1.0


def test_estimate_home_win_probability(brownian_wp):
    result = brownian_wp.estimate_home_win_probability(5, 11, 1)
    assert len(result.replications) == 1
    assert len(result.replications[0]) == 12


def test_evaluate_home_team_result():
    assert BrownianWinProbabilitySimulation._evaluate_home_team_result(5) == 1
    assert BrownianWinProbabilitySimulation._evaluate_home_team_result(-3) == 0

