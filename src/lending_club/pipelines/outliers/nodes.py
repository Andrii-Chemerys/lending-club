"""
This is a boilerplate pipeline 'outliers'
generated using Kedro 0.19.9
"""
import pandas as pd
import numpy as np
from scipy import stats

# Define function that will handle outliers applying log transformation
def outliers_handler(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    z_scores = np.abs(stats.zscore(df.select_dtypes(np.number)))
    gt_thresh = z_scores > params['threshold']
    outlier_volume = pd.DataFrame(gt_thresh.sum(), columns=['num_outliers'])
    fields_to_treat = outlier_volume[outlier_volume['num_outliers'] > 0].index.to_list()
    df[f"{fields_to_treat}_log"] = np.log1p(df[fields_to_treat].astype(float))
    return df[[f"{fields_to_treat}_log"]]