"""
This is a boilerplate pipeline 'feature_eng'
generated using Kedro 0.19.9
"""
import pandas as pd

def selected_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    
    return df[[params['target'], *params['features']]]