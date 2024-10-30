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
    print(params['none_list'])
    # Fill missing values by most frequent ones
    imp_mf = SimpleImputer(missing_values=None, strategy='most_frequent')
    df[params['freq_list']] = imp_mf.fit_transform(df[params['freq_list']])
    return df

def clean_dataset(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df1 = _drop_features(df, params)
    df2 = _fill_data(df1, params)
    return df2
