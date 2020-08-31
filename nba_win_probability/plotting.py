import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats


def histogram_plot_score_difference_by_quarter(df: pd.DataFrame):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    first_quarter = df[df['PERIOD'] == 1]['SCORE_BY_MINUTE']
    axs[0, 0].hist(first_quarter)

    axs[0, 0].set_title('1st Quarter')

    second_quarter = df[df['PERIOD'] == 2]['SCORE_BY_MINUTE']
    axs[0, 1].hist(second_quarter)
    axs[0, 1].set_title('2nd Quarter')

    third_quarter = df[df['PERIOD'] == 3]['SCORE_BY_MINUTE']
    axs[1, 0].hist(third_quarter)
    axs[1, 0].set_title('3rd Quarter')

    fourth_quarter = df[df['PERIOD'] == 4]['SCORE_BY_MINUTE']
    axs[1, 1].hist(fourth_quarter)
    _ = axs[1, 1].set_title('4th Quarter')
    plt.show()


def qq_plot_score_difference_by_quarter(df: pd.DataFrame):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    first_quarter = df[df['PERIOD'] == 1]['SCORE_BY_MINUTE']
    stats.probplot(first_quarter, dist='norm', plot=axs[0, 0])
    axs[0, 0].set_title('1st Quarter')

    second_quarter = df[df['PERIOD'] == 2]['SCORE_BY_MINUTE']
    stats.probplot(second_quarter, dist='norm', plot=axs[0, 1])
    axs[0, 1].set_title('2nd Quarter')

    third_quarter = df[df['PERIOD'] == 3]['SCORE_BY_MINUTE']
    stats.probplot(third_quarter, dist='norm', plot=axs[1, 0])
    axs[1, 0].set_title('3rd Quarter')

    fourth_quarter = df[df['PERIOD'] == 4]['SCORE_BY_MINUTE']
    stats.probplot(fourth_quarter, dist='norm', plot=axs[1, 1])
    _ = axs[1, 1].set_title('4th Quarter')
    plt.show()


def plot_reliability_diagram_by_quarter(df: pd.DataFrame):
    plt.show()