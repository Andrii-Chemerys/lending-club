"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""
import pandas as pd


def split_features_target(df: pd.DataFrame):
    y = df.default_status
    X = df.drop('default_status', axis=1)
    return X, y
