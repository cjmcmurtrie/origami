from datetime import timedelta
import pandas as pd


def fetch_nordpool(url):
    '''
    Fetch nordpool n2ex prices from url and clean them up
    :param url: str
    :return: DataFrame
    '''
    datetime_col = 'datetime'
    price_col = 'n2ex_price'
    df = pd.read_html(url)[0]
    df.columns = [col[-1] for col in df.columns.values]
    df = df.rename({
            'Unnamed: 0_level_2': datetime_col,
            'UK': price_col,
        },
        axis='columns'
    )
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[datetime_col] = df[[datetime_col, 'hours']].apply(
        lambda x:
            x[0] + timedelta(
                hours=int(x[1].split('-')[-1])
            ),
        axis=1
    )
    df = df.drop(['CET/CEST time', 'hours'], axis='columns')
    df = df.drop(0, axis=0)
    return df


def load_n2ex_price(path, datetime_col='datetime'):
    '''
    Load already cleaned nordpool data from disk. If path is a url, then fetch and clean
    :param path: url or file path
    :param datetime_col: name of time column in dataset
    :return:
    '''
    if 'html' in path:
        df = fetch_nordpool(path)
    else:
        df = pd.read_csv(path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col, ascending=True)
    return df.reset_index(drop=True)


def load_system_price(path, datetime_col='datetime'):
    '''
    Looad and clean the system prices
    :param path: file path for data
    :param datetime_col: name of time column to assign
    :return:
    '''
    price_col = 'system_price'
    net_imbalance_col = 'net_imbalance_volume'
    df = pd.read_csv(path)
    df = df.dropna(how='all', axis='rows').reset_index(drop=True)
    df = df.rename({
        'Settlement Date': datetime_col,
        'Net Imbalance Volume(MWh)': net_imbalance_col
    },
        axis='columns'
    )
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[datetime_col] = df[[datetime_col, 'Settlement Period']].apply(
        lambda x: x[0] + timedelta(hours=x[1] / 2),
        axis=1
    )
    df[price_col] = (df['System Sell Price(£/MWh)'] + df['System Buy Price(£/MWh)']) / 2
    df = df.drop(
        ['Settlement Period', 'System Sell Price(£/MWh)', 'System Buy Price(£/MWh)'],
        axis='columns'
    )
    df = df.sort_values(datetime_col, ascending=True)
    return df


def merged_syst_n2ex(syst_path, n2ex_path, datetime_col='datetime'):
    '''
    Get joined system prices and n2ex prices, joined by closest backward timestamp
    :param syst_path: system prices file path
    :param n2ex_path: n2ex prices file path or url
    :param datetime_col: name of time column
    :return: DataFrame
    '''
    syst = load_system_price(syst_path)
    n2ex = load_n2ex_price(n2ex_path)
    return pd.merge_asof(
        syst, n2ex,
        on=datetime_col,
        direction='backward'
    )
