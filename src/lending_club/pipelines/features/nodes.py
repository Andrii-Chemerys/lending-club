"""
This is a boilerplate pipeline 'feature_eng'
generated using Kedro 0.19.9
"""
import pandas as pd
import numpy as np

# def selected_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
#TODO: Remove    
#     return df[[params['target'], *params['features']]]

# Function that will select features with strong correlation
# to the target based on two thresholds feeded as parameters: 
#   - correlation to target
#   - intercorellation between features
def selected_features(df: pd.DataFrame, params: dict) -> dict:
    eda_corr=df.corr()
    def_stat_corr = eda_corr.default_status.sort_values().drop('default_status')
    # Compute a correlation matrix and convert to long-form
    target_high_corr = def_stat_corr[np.abs(def_stat_corr) > params['corr_treshold']]
    corr_mtx = df[target_high_corr.index.to_list()].corr().stack().reset_index(name='Correlation')
    df_high_corr = corr_mtx[(corr_mtx.Correlation != 1) & (corr_mtx.Correlation > params['intercorr_treshold'])].sort_values(by='Correlation', ascending=False).drop_duplicates('Correlation')
    # Select important variables for prediction model
    important_feat = {
        'target': 'default_status',
        'model_features': list(set(target_high_corr.index.to_list()) - set(df_high_corr['level_0']))
    }
    return important_feat