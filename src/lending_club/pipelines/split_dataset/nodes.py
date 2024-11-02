"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger(__name__)

def split_balance(df: pd.DataFrame, params: dict):
    y = df[['default_status']]
    X = df.drop('default_status', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['test_size'], 
        random_state=params['random_state']
    )
    logger.info("Train imbalanced datasets size (X, y): %d, %d", X_train.shape[0], y_train.shape[0])
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train) # type: ignore
    logger.info("Train balanced datasets size (X, y): %d, %d", X_train.shape[0], y_train.shape[0])
    return X_train, X_test, y_train, y_test
