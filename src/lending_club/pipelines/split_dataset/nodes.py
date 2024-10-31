"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""
from sklearn.model_selection import train_test_split
import pandas as pd

def _train_test_split():
    ...

def _split_features_target(df: pd.DataFrame):
    y = df.default_status
    X = df.drop('default_status', axis=1)
    return X, y