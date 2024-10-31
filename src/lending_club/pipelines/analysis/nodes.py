"""
This is a boilerplate pipeline 'analysis'
generated using Kedro 0.19.9
"""
import pandas as pd

# Function for merging features for analisys from clean and encoded datasets
def eda_df (df_clean: pd.DataFrame, df_encode: pd.DataFrame, params: dict) -> pd.DataFrame:
    return pd.concat([df_clean[params['clean']], df_encode[params['encoded']]], axis=1)
