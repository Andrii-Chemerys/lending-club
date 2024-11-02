"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# def train_model(X_train, y_train, params: dict):
#     cb = CatBoostClassifier(
#             iterations=params['iterations'],						# Maximum number of boosting iterations
#             learning_rate=params['learning_rate'],					# Learning rate
#             eval_metric=params['eval_metric'],					    # Metric to monitor
#             custom_loss=params['custom_loss'],                      # Additional metrics to plot
#             early_stopping_rounds=params['early_stopping_rounds'],	# Stop if no improvement after X iterations
#             od_type=params['od_type'],						        # Overfitting detection type (detect after fixed number of non-improving iterations)
#             random_seed=params['random_seed'],
#             verbose=params['verbose'],						        # Print log every X iterations
#             eval_fraction=params['eval_fraction']                   # Fraction of training dataset for validation
#     )    

#     cb.fit(X_train, y_train,
#            use_best_model=True,
#            plot=True)
#     return cb

def evaluate_metrics(model: object, name: str, X_true: object, y_true: object) -> pd.DataFrame:
    y_pred = model.predict(X_true)
    # NN model return probabilities that we should translate to
    # True/False based on treshold = 0.5
    if not isinstance(y_pred, list):
        y_pred = (y_pred > 0.5)
    metrics = pd.DataFrame()
    metrics['accuracy']  = {name: accuracy_score(y_true, y_pred)}
    metrics['precision'] = {name: precision_score(y_true, y_pred)}
    metrics['recall']    = {name: recall_score(y_true, y_pred)}
    metrics['f1']        = {name: f1_score(y_true, y_pred)}
    metrics['roc_auc']   = {name: roc_auc_score(y_true, y_pred)}
    metrics['conf_mtx']  = {name: confusion_matrix(y_true, y_pred)}
    print("Model's scores:\n", metrics)
    
'''
Merging in one pipeline functions from:
    - data_clean pipeline
    - encode pipeline
'''

# Function to encode emp_length to number
def _parse_emp_len(df: pd.DataFrame, emp_len: str) -> pd.DataFrame:
    df[emp_len] = df[emp_len].str.split(" ").str[0].str.replace("+", "").str.replace("<", "0").astype(int)
    return df[emp_len]

# Function to encode default_status from loan_status, i.e. our target
def _default_status(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df['default_status'] = (
            (df[params['default']] == 'Charged Off') |
            (df[params['default']] == 'Does not meet the credit policy. Status:Charged Off') |
            (df[params['default']] == 'Default')
    )
    return df[['default_status']]

def model_pipeline(df: pd.DataFrame, params: dict):
    # Drop duplicates and selected features
    df.drop_duplicates(inplace=True)
    df.drop(params["drop_list"], axis=1, inplace=True)
    # pipelines/data_clean replacement
    data_clean = make_column_transformer(
        (SimpleImputer(missing_values=None, strategy='constant', fill_value='none'), df[params['none_list']]),
        (SimpleImputer(missing_values=None, strategy='most_frequent'), df[params['freq_list']]),
        (SimpleImputer(strategy='constant', fill_value=0), df[params['fill_zero']]),
        (SimpleImputer(strategy='median'), df[params['fill_med']]),
        remainder='passthrough'
    )
    # pipelines/encode replacement
    encode = make_column_transformer(
        (OrdinalEncoder(), df[params['category']]),
        (FunctionTransformer(_parse_emp_len), params['emp_len']),
        # TODO: Must be done before splitting
        # (FunctionTransformer(_default_status), params['default']), 
        remainder='passthrough'
    )
    pipeline = make_pipeline([
        data_clean,
        encode,
        StandardScaler(),
        CatBoostClassifier(
            iterations=params['iterations'],						# Maximum number of boosting iterations
            learning_rate=params['learning_rate'],					# Learning rate
            eval_metric=params['eval_metric'],					    # Metric to monitor
            custom_loss=params['custom_loss'],                      # Additional metrics to plot
            early_stopping_rounds=params['early_stopping_rounds'],	# Stop if no improvement after X iterations
            od_type=params['od_type'],						        # Overfitting detection type (detect after fixed number of non-improving iterations)
            random_seed=params['random_seed'],
            verbose=params['verbose'],						        # Print log every X iterations
            eval_fraction=params['eval_fraction']                   # Fraction of training dataset for validation
        )   
    ])
    return pipeline