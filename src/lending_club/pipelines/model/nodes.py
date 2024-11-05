"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import logging
from ..encode.nodes import _default_status

logger = logging.getLogger(__name__)

def split_n_balance(df: pd.DataFrame, df_fe: pd.DataFrame, params: dict):
    y = _default_status(df, params)
    X = pd.concat([df, df_fe], axis=1)
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


def _drop_duplicates(df: pd.DataFrame):
    return df.drop_duplicates()


def model_pipeline(params: dict):

    # split important features to assign preprocessing steps
    category_feat = [f for f in (params['category'] + params['emp_len']) if f in params['model_features']]
    numeric_feat_zero = [f for f in params['fill_zero'] if f in params['model_features']]
    numeric_feat_med = [f for f in params['fill_med'] if f in params['model_features']]

    # transformer to replace missing numeric values by 0
    numeric_feat_zero_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=0),
        StandardScaler()
    )
    # transformer to replace missing numeric values by median
    numeric_feat_med_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    # assemble transformers to preprocessing pipe
    preprocessing = make_column_transformer(
        (OrdinalEncoder(), category_feat),
        (numeric_feat_zero_transformer, numeric_feat_zero),
        (numeric_feat_med_transformer, numeric_feat_med)
    )

    regressor_params = params['model_options']['regressor_options']

    pipeline = make_pipeline(
        preprocessing,
        # SMOTE(random_state=params['random_state']),
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
