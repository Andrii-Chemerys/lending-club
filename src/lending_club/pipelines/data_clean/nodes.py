"""
This is a boilerplate pipeline 'data_clean'
generated using Kedro 0.19.9
"""
import pandas as pd
from sklearn.impute import SimpleImputer

def _drop_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    return df.drop(params["drop_list"], axis=1)


def _fill_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:

    # Fill missing values by 'None'
    imp_none = SimpleImputer(missing_values=None, strategy='constant', fill_value='none')
    df[params['none_list']] = imp_none.fit_transform(df[params['none_list']])
    
    # Fill missing values by most frequent ones
    imp_mf = SimpleImputer(missing_values=None, strategy='most_frequent')
    df[params['freq_list']] = imp_mf.fit_transform(df[params['freq_list']])

    # Fill missing values by 0
    imp_zer = SimpleImputer(strategy='constant', fill_value=0)
    df[params['fill_zero']] = imp_zer.fit_transform(df[params['fill_zero']])

    # Fill missing values by median
    imp_med = SimpleImputer(strategy='median')
    df[params['fill_med']] = imp_med.fit_transform(df[params['fill_med']])
    
    return df


def clean_dataset(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = _drop_features(df, params)
    df = _fill_data(df, params)
    return df
