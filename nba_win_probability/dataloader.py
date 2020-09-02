import os
import pandas as pd

from typing import List

AVAILABLE_SEASONS = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]


def load_season(season: str) -> pd.DataFrame:
    if season not in AVAILABLE_SEASONS:
        raise ValueError(f"Data for season {season} is not available. Try the following: {AVAILABLE_SEASONS}")

    path = f"../pbp_data/{season}_pbp.csv"
    data = pd.read_csv(path)
    data['SEASON'] = season
    return data


def load_multiple_seasons(seasons: List[str]):
    data = [load_season(season) for season in seasons]
    return pd.concat(data)