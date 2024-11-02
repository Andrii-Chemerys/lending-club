"""
This is a boilerplate pipeline 'feature_eng'
generated using Kedro 0.19.9
"""
import pandas as pd

def selected_features(df_prim: pd.DataFrame, 
                      df_fe: pd.DataFrame, 
                      params: dict) -> pd.DataFrame:
    df = df_prim[params['features']]
    return pd.concat([df, 
                      df_fe if params['ignore_new_features'] else None], 
                      axis=1)
