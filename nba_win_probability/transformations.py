import pandas as pd
import numpy as np

from nba_win_probability.win_probability import BrownianWinProbabilitySimulation


def transform_data_for_analysis(df: pd.DataFrame):
    """Prepares data for modelling procedure by calculating time information and
        the score difference by minute.
    """
    _df = df.copy()
    return (_df.pipe(add_time_information)
            .pipe(adjust_score_margin)
            .pipe(calculate_score_difference_by_minute))


def add_time_information(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new dataframe and adds columns for quarter timestamp, overall timestamp, and minute"""
    _df = df.copy()

    # Remove beginning of non-first quarter because it is already accounted for at end of quarters
    _df = _df[(_df['PCTIMESTRING'] != '12:00') | (_df['PERIOD'] == 1)]

    # Use simple calculations to define quarter timestamp and overall timestamp
    _df['QUARTER_TS'] = _df['PCTIMESTRING'].str.split(':').apply(lambda x: 12 - int(x[0]) - (int(x[1]) / 60))
    _df['TIME_ELAPSED'] = _df['QUARTER_TS'] + (_df['PERIOD'] - 1) * 12
    _df['MINUTE'] = np.floor(_df['TIME_ELAPSED'])

    return _df


def adjust_score_margin(df: pd.DataFrame) -> pd.DataFrame:
    """This function fixes the NaN values in the SCOREMARGIN column by giving them their proper values.

        Additionally, this function casts SCOREMARGIN to a float instead of an object.
    """
    _df = df.copy()
    _df.loc[(_df['PCTIMESTRING'] == '12:00') & (_df['PERIOD'] == 1), 'SCOREMARGIN'] = 0
    _df = _df[_df['SCOREMARGIN'].notnull()]
    _df.loc[_df.SCOREMARGIN == 'TIE', 'SCOREMARGIN'] = 0

    _df = _df.astype({'SCOREMARGIN': float})

    return _df


def calculate_score_difference_by_minute(df: pd.DataFrame) -> pd.DataFrame:
    """This function calculates the minute-by-minute changes in the score margin"""
    _df = df.copy()

    # Find last event from each minute of each game
    _df.sort_values(by=['GAME_ID', 'TIME_ELAPSED'], inplace=True)
    _df = _df.groupby(by=['GAME_ID', 'MINUTE']).last().reset_index()

    # Take the difference of scores between each minute
    _df['SCORE_BY_MINUTE'] = _df.groupby('GAME_ID')['SCOREMARGIN'].diff().fillna(0)

    return _df


def add_game_result_column(df: pd.DataFrame) -> pd.DataFrame:
    """Determines the game winner by checking the score differential at the end of the game"""
    _df = df.copy()

    _df['RESULT'] = _df.apply(lambda row: determine_game_result(row['SCOREMARGIN']) if (
            row['PCTIMESTRING'] == "0:00" and row['PERIOD'] == 4) else 'undefined', axis=1)
    _df.sort_values(by=['GAME_ID', 'TIME_ELAPSED'], inplace=True)
    _df['RESULT'] = _df.groupby(['GAME_ID'])['RESULT'].transform('last')

    return _df


def determine_game_result(final_score_margin: int):
    """Determines the game result by using the final scoring margin in a game."""
    if final_score_margin > 0:
        return 1
    elif final_score_margin < 0:
        return 0
    else:
        return 'undefined'


def get_moment_from_each_game(df: pd.DataFrame):
    """Takes a random data point from each unique GAME_ID in the dataframe."""
    _df = df.copy()
    _df = _df[_df['TIME_ELAPSED'] < 48].groupby('GAME_ID').apply(lambda x: x.sample(1)).reset_index(drop=True)
    return _df


def assign_win_probabilities(df: pd.DataFrame, brownian_win_probability: BrownianWinProbabilitySimulation):
    """Assigns the estimated win probability to each row."""
    _df = df.copy()
    _df['PROBA'] = _df.apply(lambda row: get_win_probability(row, brownian_win_probability) ,axis=1)
    return _df


def get_win_probability(row, brownian_win_probability):
    """Estimates the win probability of a row by running a simulation based on the time remaining
        and the score margin.
    """
    time_remaining = get_time_remaining(int(row['MINUTE']))
    result = brownian_win_probability.estimate_home_win_probability(row['SCOREMARGIN'], time_remaining)
    return result.estimated_win_probability


def get_time_remaining(time_elapsed: float):
    """Calculates the time remaining in a 48 minute game."""
    time_remaining = 48 - time_elapsed
    return time_remaining
