import pandas as pd


def forecast_target(min_fall, max_rise, negative_threshold, positive_threshold):
    '''
    The logic for building the training labels, which is as follows:
        - if the price rose above the positive_threshold but did not fall below the negative, label 'positive'
        - if the price fell below the negative_threshold but did not rise above the positive, label 'negative'
        - if the price went past both thresholds, label 'volatile'
        - if the price did not go past either threshold, label 'stable'
        - otherwise label 'unknown' (for nan cases in the data)
    :param min_fall: the largest price fall in the time window
    :param max_rise: the largest price rise in the time window
    :param negative_threshold: the negative threshold for assigning labels
    :param positive_threshold: the positive thrshold for assigning labels
    :return: str
    '''
    if min_fall > negative_threshold and max_rise >= positive_threshold:
        return 'positive'
    elif min_fall <= negative_threshold and max_rise < positive_threshold:
        return 'negative'
    elif min_fall <= negative_threshold and max_rise >= positive_threshold:
        return 'volatile'
    elif min_fall > negative_threshold and max_rise < positive_threshold:
        return 'stable'
    else:
        return 'unknown'


def construct_target(df, time_col, price_col, horizon_periods, positive_threshold, negative_threshold):
    '''
    Construct a multi-class target variable for forecasting. The future time-window window with length equal
    to horizon_periods is scanned. Depending on the price behaviour in that window, one of 'positive', 'negative',
    'volatile' or 'stable' is assigned to each row.
    :param df: DataFrame
    :param time_col: the name of the time column
    :param price_col: thee price column to build a target around
    :param horizon_periods: the length of the time window to consider for forecasting
    :param positive_threshold: see the forecast_target doc string
    :param negative_threshold: see the forecast_target doc string
    :return: DataFrame
    '''
    df[price_col] = df[price_col].ffill()
    rolled = df.set_index(time_col) \
        .shift(-1) \
        .rolling(horizon_periods)
    rolled_min = rolled.min() \
        .reset_index(drop=False) \
        .shift(-(horizon_periods - 1))
    rolled_max = rolled.max() \
        .reset_index(drop=False) \
        .shift(-(horizon_periods - 1))
    min_ahead_col = 'min_price_{}_ahead'.format(horizon_periods)
    max_ahead_col = 'max_price_{}_ahead'.format(horizon_periods)
    df = pd.concat([
            df,
            rolled_min.rename({price_col: min_ahead_col}, axis='columns')[min_ahead_col],
            rolled_max.rename({price_col: max_ahead_col}, axis='columns')[max_ahead_col]
        ],
        axis='columns'
    )
    df['min_fall_percent'] = df[[price_col, min_ahead_col]].apply(
        lambda x: (x[1] / x[0]) - 1,
        axis=1
    )
    df['max_rise_percent'] = df[[price_col, max_ahead_col]].apply(
        lambda x: (x[1] / x[0]) - 1,
        axis=1
    )
    df['forecast'] = df[['min_fall_percent', 'max_rise_percent']].apply(
        lambda x: forecast_target(
                min_fall=x[0],
                max_rise=x[1],
                negative_threshold=negative_threshold,
                positive_threshold=positive_threshold
        ),
        axis=1
    )
    df = df[df['forecast'] != 'unknown']
    df = df.drop(
        [min_ahead_col, max_ahead_col, 'min_fall_percent', 'max_rise_percent'],
        axis='columns'
    )
    return df
