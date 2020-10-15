

def add_seasonal_features(df, time_col, group_col=None):
    '''
    Add some seasonal features to the DataFrame, based on time_col
    :param df: DataFrame
    :param time_col: name of time column
    :param group_col: todo
    :return: DataFrame
    '''
    # todo: add `season` col
    # todo: add `is_weekend` col
    df['date'] = df[time_col].dt.date
    df['hour'] = df[time_col].dt.hour
    df['dayofweek'] = df[time_col].dt.dayofweek
    df['month'] = df[time_col].dt.month
    feature_names = ['hour', 'month', 'dayofweek']
    return df, feature_names


def normalise_feature_values(df, feature_columns):
    '''
    Normalise the feature_columns to values between 0 and 1
    :param df: DataFrame
    :param feature_columns: names of the columns to normalisee
    :return:  DataFrame
    '''
    df[feature_columns] = \
        (df[feature_columns] - df[feature_columns].min()) / \
        (df[feature_columns].max() - df[feature_columns].min())
    return df


def impute_missing(df, feature_columns):
    # todo: rolling median imputation
    '''
    Impute missing values (for now, just with zero)
    :param df: DataFrame
    :param feature_columns: names of columns to impute
    :return: DataFrame
    '''
    df[feature_columns] = df[feature_columns].fillna(0)
    return df
