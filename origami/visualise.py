import calendar
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns


def column_compare_plot(df, group_column, comparison_columns, time_dim=None):
    '''
    Plot a group of normalised columns (with outliers removed) to compare
    :param df: input dataframe
    :param group_column: the group column for the x-axis
    :param comparison_columns: list of columns to compare
    :param time_dim: optional parameter to compare on the time dimension
    :return:
    '''
    sns.set(rc={'figure.figsize':(12, 4)})
    df = df.copy()
    if isinstance(time_dim, str):
        df = df.set_index(group_column).groupby(pd.Grouper(freq=time_dim)).mean().reset_index()
    dfs = []
    for comp in comparison_columns:
        sub_df = df[[group_column, comp]]
        sub_df = sub_df.rename({comp: 'value'}, axis='columns')
        sub_df = sub_df[(np.abs(sp.stats.zscore(sub_df['value'])) < 3)]
        sub_df['value'] /= sub_df['value'][0]
        sub_df['name'] = comp
        dfs.append(
            sub_df
        )
    df = pd.concat(dfs, axis=0)
    sns.lineplot(data=df, x=group_column, y="value", hue="name", ci='sd')


def time_of_day_plot(df, time_column, agg_column):
    '''
    Plot the average values of a column by time of day
    :param df: DataFrame
    :param time_column: time column name
    :param agg_column: plot column name
    :return:
    '''
    sns.set(rc={'figure.figsize':(12, 4)})
    df = df.copy()
    df['hour'] = df[time_column].dt.hour
    sns.lineplot(data=df, x='hour', y=agg_column, ci="sd")


def month_plot(df, time_column, agg_column):
    '''
    Plot the average values of a column by month
    :param df: DataFrame
    :param time_column: time column name
    :param agg_column: plot column name
    :return:
    '''
    sns.set(rc={'figure.figsize':(12, 4)})
    df = df.copy()
    df['month'] = df[time_column].dt.month.apply(
        lambda x: calendar.month_abbr[x]
    )
    sns.lineplot(data=df, x='month', y=agg_column, ci="sd")


def day_of_week_plot(df, time_column, agg_column):
    '''
    Plot the average values of a column by day of week
    :param df: DataFrame
    :param time_column: time column name
    :param agg_column: plot column name
    :return:
    '''
    sns.set(rc={'figure.figsize':(12, 4)})
    df = df.copy()
    df['dayofweek'] = df[time_column].dt.dayofweek.apply(
        lambda x: calendar.day_name[x]
    )
    sns.lineplot(data=df, x='dayofweek', y=agg_column, ci=None)
