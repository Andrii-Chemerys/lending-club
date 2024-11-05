"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import logging
from ..analysis.nodes import features_eng
from ..encode.nodes import _default_status

logger = logging.getLogger(__name__)

def split_dataset(df: pd.DataFrame, params: dict):
    y = _default_status(df, params)
    X = df #TODO Remove .drop('default_status', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['test_size'],
        random_state=params['random_state']
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, regressor, params: dict):
    try:
        regressor.set_params(**params['fit_options']).fit(X_train, y_train) 
    except:
        regressor.fit(X_train, y_train)
    return regressor

def evaluate_metrics(model: object, params: dict, X_true: object, y_true: object) -> pd.DataFrame:
    y_pred = model.predict(X_true)
    # NN model return probabilities that we should translate to
    # True/False based on treshold = 0.5
    if not isinstance(y_pred, list):
        y_pred = (y_pred > 0.5)
    metrics = pd.DataFrame()
    metrics['accuracy']  = {params['name']: accuracy_score(y_true, y_pred)}
    metrics['precision'] = {params['name']: precision_score(y_true, y_pred)}
    metrics['recall']    = {params['name']: recall_score(y_true, y_pred)}
    metrics['f1']        = {params['name']: f1_score(y_true, y_pred)}
    metrics['roc_auc']   = {params['name']: roc_auc_score(y_true, y_pred)}
    metrics['conf_mtx']  = {params['name']: confusion_matrix(y_true, y_pred)}
    print("Model's scores:\n", metrics)



'''
Merging in one pipeline functions from:
    - data_process pipeline
    - data_clean pipeline
    - encode pipeline
'''

def _drop_duplicates(df: pd.DataFrame):
    return df.drop_duplicates()

# Function to encode emp_length to number
def _parse_emp_len(df: pd.DataFrame, emp_len: str) -> pd.DataFrame:
    df[emp_len] = df[emp_len].str.split(" ").str[0].str.replace("+", "").str.replace("<", "0").astype(int)
    return df[[emp_len]]

def model_pipeline(params: dict):
    # Drop duplicates and selected features
    drop_trash = make_column_transformer(
        ('drop', params["drop_list"]),
        remainder=FunctionTransformer(_drop_duplicates)
    )
    # pipelines/data_clean replacement
    data_clean = make_column_transformer(
        (SimpleImputer(missing_values=None, strategy='constant', fill_value='none'), params['none_list']),
        (SimpleImputer(missing_values=None, strategy='most_frequent'), params['freq_list']),
        (SimpleImputer(strategy='constant', fill_value=0), params['fill_zero']),
        (SimpleImputer(strategy='median'), params['fill_med']),
        remainder='passthrough'
    )
    # pipelines/encode replacement
    encode = make_column_transformer(
        (OrdinalEncoder(), params['category']),
        (FunctionTransformer(_parse_emp_len), params['emp_len']),
        remainder='passthrough'
    )
    # feature engineering node replacement
    new_features = make_column_transformer(
        (FunctionTransformer(features_eng), []),
        remainder='passthrough'
    )
    # scale features selected for model and drop the rest
    normalizer = make_column_transformer(
        (StandardScaler(), params['model_features']),
        remainder='drop'
    )
    regressor_params = params['model_options']['regressor_options']
    pipeline = make_pipeline(
        drop_trash,
        data_clean,
        encode,
        new_features,
        normalizer,
        SMOTE(random_state=params['random_state']),
        # CatBoostClassifier(
        #     iterations=regressor_params['iterations'],						# Maximum number of boosting iterations
        #     learning_rate=regressor_params['learning_rate'],					# Learning rate
        #     eval_metric=regressor_params['eval_metric'],					    # Metric to monitor
        #     custom_loss=regressor_params['custom_loss'],                      # Additional metrics to plot
        #     early_stopping_rounds=regressor_params['early_stopping_rounds'],	# Stop if no improvement after X iterations
        #     od_type=regressor_params['od_type'],						        # Overfitting detection type (detect after fixed number of non-improving iterations)
        #     random_seed=regressor_params['random_seed'],
        #     verbose=regressor_params['verbose'],						        # Print log every X iterations
        #     eval_fraction=regressor_params['eval_fraction']                   # Fraction of training dataset for validation
        # )
        RandomForestClassifier(regressor_params)
    )
    return pipeline
