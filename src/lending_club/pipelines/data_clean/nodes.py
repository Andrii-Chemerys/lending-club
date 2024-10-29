"""
This is a boilerplate pipeline 'data_clean'
generated using Kedro 0.19.9
"""
import pandas as pd

# Define function to drop unnecessary features
def _drop_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    return df.drop(params["drop_list"], axis=1)




def clean_dataset(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = _drop_features(df, params)
    return df