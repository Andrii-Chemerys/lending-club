"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""

import logging
import pandas as pd
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from ..encode.nodes import _default_status

logger = logging.getLogger(__name__)

def split_dataset(df: pd.DataFrame, params: dict):
    y = _default_status(df, params)
    X = df
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


def make_rng(start, stop, step):
    return range(start, stop, step)


def evaluate_metrics(model: object, X_true, y_true,
                     params: dict) -> pd.DataFrame:
    y_pred_proba = model.predict_proba(X_true)
    metrics = pd.DataFrame()
    for thresh in make_rng(**params['model_options']['prob_threshold']):
        y_pred = (y_pred_proba[:,1] > (thresh / 100))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cur_metrics = pd.DataFrame(
        data={
            'prob_tresh_%': thresh,
            'accuracy'    : accuracy_score(y_true, y_pred),
            'precision'   : precision_score(y_true, y_pred),
            'recall'      : recall_score(y_true, y_pred),
            'f1'          : f1_score(y_true, y_pred),
            'roc_auc'     : roc_auc_score(y_true, y_pred),
            'tn'          : tn,
            'fp'          : fp,
            'fn'          : fn,
            'tp'          : tp,
            'loss'        : params['FP_cost'] * fp + params['FN_cost'] *fn,
        },
        index = [params['model_options']['name']]
        )
        metrics = pd.concat([metrics, cur_metrics], axis=0)
    logger.info(f"The best probability threshold for {params['model_options']['name']} model based on min loss: {metrics[metrics.loss==metrics.loss.min()]['prob_thresh_%'].iloc[0]}")
    return metrics


def _drop_duplicates(df: pd.DataFrame):
    return df.drop_duplicates()


def model_pipeline(model_options: dict, params: dict):

    set_config(transform_output='pandas')

    # split important features to assign preprocessing steps
    category_feat = [f for f in (params['category'] + [params['emp_len']]) if f in params['model_features']]
    numeric_feat_zero = [f for f in params['fill_zero'] if f in params['model_features']]
    numeric_feat_med = [f for f in params['fill_med'] if f in params['model_features']]
    remainder_feat = list(set(params['model_features']) - set(category_feat) - set(numeric_feat_zero) - set(numeric_feat_med))

    # transformer to replace missing numeric values by 0
    # and standardize all values 
    numeric_feat_zero_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=0),
        StandardScaler()
    )
    # transformer to replace missing numeric values by median
    numeric_feat_med_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    # assemble transformers in preprocessing pipe so it will perform 
    # following transformations:
    #   - encode all categorical features to numbers
    #   - fill missing values in specific number features as "0" and standardize them
    #   - fill missing values in specific number features as median and standardize them
    #   - standardize the rest of the features
    preprocessing = make_column_transformer(
        (OrdinalEncoder(), category_feat),
        (numeric_feat_zero_transformer, numeric_feat_zero),
        (numeric_feat_med_transformer, numeric_feat_med),
        (StandardScaler(), remainder_feat)
    )

    # choose regressor depending on provided model_options
    if model_options['name'] == 'rfc':
        regressor = RandomForestClassifier(**model_options['regressor_options'])
    else: 
        if model_options['name'] == 'catboost':
            regressor = CatBoostClassifier(**model_options['regressor_options'])
        else:
            raise Exception("Pipeline accepts only RandomForestClassifier and CatBoostClassifier")
    
    # Assemble preprocessing pipeline, imbalance handling and chosen regressor as the model pipeline
    model = imb_make_pipeline(
        preprocessing,
        SMOTE(random_state=params['random_state']),
        regressor
    )
    return model
