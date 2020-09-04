# nba-win-probability
NBA Win Probability model

## Contents

This repo contains code for modeling NBA win probability and various notebooks for exploration and analysis
of the modeling approach.

#### Code

- `dataloader.py`: functions for loading play-by-play data into pandas
- `plotting.py`: functions for displaying specific plots inside of notebooks
- `transformations.py`: dataframe transformations required for building model
- `win_probability`: code for win_probability model

#### Notebooks

- `Part 1 - ValidateNormality`: notebook for confirming model assumptions of normality in the data
- `Part 2 - ModelFit`: notebook showing basic model usage and results
- `Part 3 - EvaluateModel`: model evaluation code comparing training and testing code

## Data Sources
- https://eightthirtyfour.com/data

## Papers Referenced
- [A Brownian Motion Model for the Progress of Sports Scores](https://www.stat.berkeley.edu/~aldous/157/Papers/stern.pdf)