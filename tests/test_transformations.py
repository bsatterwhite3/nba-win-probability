import pytest

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from nba_win_probability import transformations, win_probability
from nba_win_probability.win_probability import WinProbabilitySimulation


def test_add_time_information():
    data = {'PCTIMESTRING': ['12:00', '11:00', '5:00'], 'PERIOD': [2, 3, 1]}
    df = pd.DataFrame(data)

    result = transformations.add_time_information(df)

    expected_df = pd.DataFrame({
        'PCTIMESTRING': ['11:00', '5:00'],
        'PERIOD': [3, 1],
        'QUARTER_TS': [1.0, 7.0],
        'TIME_ELAPSED': [25.0, 7.0],
        'MINUTE': [25.0, 7.0]
    }, index=[1, 2])

    assert_frame_equal(result, expected_df)


def test_adjust_score_margin():
    df = pd.DataFrame({
        'PCTIMESTRING': ['12:00', '11:00', '5:00'],
        'PERIOD': [1, 2, 3],
        'SCOREMARGIN': [np.NaN, 'TIE', 14]
    })

    expected_df = pd.DataFrame({
        'PCTIMESTRING': ['12:00', '11:00', '5:00'],
        'PERIOD': [1, 2, 3],
        'SCOREMARGIN': [0.0, 0.0, 14.0]
    }, index=[0, 1, 2])

    result = transformations.adjust_score_margin(df)

    assert_frame_equal(result, expected_df)


def test_calculate_score_difference_by_minute():
    df = pd.DataFrame({
        'GAME_ID': ['1', '1', '2'],
        'TIME_ELAPSED': [20, 19, 26],
        'MINUTE': [20, 19, 14],
        'SCOREMARGIN': [13, 10, 5]
    })

    expected_df = pd.DataFrame({
        'GAME_ID': ['1', '1', '2'],
        'MINUTE': [19, 20, 14],
        'TIME_ELAPSED': [19, 20, 26],
        'SCOREMARGIN': [10, 13, 5],
        'SCORE_BY_MINUTE': [0.0, 3.0, 0.0]
    }, index=[0, 1, 2])

    result = transformations.calculate_score_difference_by_minute(df)

    assert_frame_equal(result, expected_df)


def test_transform_data_for_analysis():
    df = pd.DataFrame({
        'GAME_ID': ['1', '1', '2'],
        'PCTIMESTRING': ['11:30', '11:00', '5:00'],
        'PERIOD': [3, 3, 1],
        'SCOREMARGIN': ['7.0', '5.0', 'TIE']
    })

    expected_df = pd.DataFrame({
        'GAME_ID': ['1', '1', '2'],
        'MINUTE': [24.0, 25.0, 7.0],
        'PCTIMESTRING': ['11:30', '11:00', '5:00'],
        'PERIOD': [3, 3, 1],
        'SCOREMARGIN': [7.0, 5.0, 0.0],
        'QUARTER_TS': [0.5, 1.0, 7.0],
        'TIME_ELAPSED': [24.5, 25, 7.0],
        'SCORE_BY_MINUTE': [0.0, -2.0, 0.0]
    })

    result = transformations.transform_data_for_analysis(df)
    assert_frame_equal(result, expected_df)


def test_add_game_result_column():
    df = pd.DataFrame({
        'GAME_ID': ['1', '1', '2', '2', '3'],
        'SCOREMARGIN': [5, 7, -1, 0, 25],
        'PCTIMESTRING': ['0:30', '0:00', '1:03', '0:00', '12:00'],
        'PERIOD': [3, 4, 2, 4, 4],
        'TIME_ELAPSED': [35.5, 48, 23, 48, 36]
    })

    expected_df = pd.DataFrame({
        'GAME_ID': ['1', '1', '2', '2', '3'],
        'SCOREMARGIN': [5, 7, -1, 0, 25],
        'PCTIMESTRING': ['0:30', '0:00', '1:03', '0:00', '12:00'],
        'PERIOD': [3, 4, 2, 4, 4],
        'TIME_ELAPSED': [35.5, 48.0, 23.0, 48.0, 36.0],
        'RESULT': [1, 1, 'undefined', 'undefined', 'undefined']
    }, index=[0, 1, 2, 3, 4])

    result = transformations.add_game_result_column(df)

    assert_frame_equal(result, expected_df)


def test_determine_game_result():
    assert transformations.determine_game_result(5) == 1
    assert transformations.determine_game_result(-7) == 0
    assert transformations.determine_game_result(0) == 'undefined'


def test_get_moment_from_each_game():
    df = pd.DataFrame({
        'GAME_ID': ['1', '1', '2', '2', '3'],
        'TIME_ELAPSED': [35.5, 48, 23, 48, 36]
    })

    result = transformations.get_moment_from_each_game(df)
    assert len(result) == 3
    assert 48.0 not in result['TIME_ELAPSED'].values


def test_assign_win_probabilities(mocker):
    bwp = win_probability.BrownianWinProbabilitySimulation(pd.Series([1]))
    mocker.patch.object(bwp, 'estimate_home_win_probability',
                        return_value=WinProbabilitySimulation.WinProbabilityOutcome(estimated_win_probability=0.75,
                                                                                    replications=[]))

    df = pd.DataFrame({
        'MINUTE': [27],
        'SCOREMARGIN': [7]
    })

    expected_df =  pd.DataFrame({
        'MINUTE': [27],
        'SCOREMARGIN': [7],
        'PROBA': [0.75]
    })

    result = transformations.assign_win_probabilities(df, bwp)
    assert_frame_equal(result, expected_df)


def test_get_win_probability(mocker):
    bwp = win_probability.BrownianWinProbabilitySimulation(pd.Series([1]))
    mocker.patch.object(bwp, 'estimate_home_win_probability',
                        return_value=WinProbabilitySimulation.WinProbabilityOutcome(estimated_win_probability=0.75,
                                                                                    replications=[]))

    row = pd.Series([27, 5], index=['MINUTE', 'SCOREMARGIN'])
    result = transformations.get_win_probability(row, bwp)
    assert result == 0.75


def test_get_time_remaining():
    time_remaining = transformations.get_time_remaining(12.5)
    assert time_remaining == 35.5
