"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def split_features(df: pd.DataFrame, params: dict):
    y = df.default_status
    X = df.drop('default_status', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['test_size'], 
        random_state=params['random_state']
    )
    return X_train, X_test, y_train, y_test

def standartizer(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def balancer(X_train, y_train):
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train) # type: ignore
    return X_train, y_train

def train_model(X_train, y_train, params: dict):
    cb = CatBoostClassifier(
            params['iterations'],						# Maximum number of boosting iterations
            params['learning_rate'],					# Learning rate
            params['eval_metric'],					    # Metric to monitor
            params['custom_loss'],                      # Additional metrics to plot
            params['early_stopping_rounds'],			# Stop if no improvement after X iterations
            params['od_type'],						    # Overfitting detection type (detect after fixed number of non-improving iterations)
            params['random_seed'],
            params['verbose'],						    # Print log every X iterations
            params['eval_fraction']                     # Fraction of training dataset for validation
    )    

    cb.fit(X_train, y_train,
           use_best_model=True,
           plot=True)
    return cb

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
    return(metrics)