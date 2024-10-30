"""
This is a boilerplate pipeline 'feature_eng'
generated using Kedro 0.19.9
"""
import pandas as pd

# Define dates feature engineering function
def _dates_fe(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    # Feature engineering dataset will be new dataframe
    fe_df = pd.DataFrame()
    # Calculate how many months ago the earliest credit line was open
    fe_df['mo_since_earliest_cr_line'] = (
        (pd.Timestamp(params['cur_date']) - df.earliest_cr_line) / pd.Timedelta(30, 'D')
    ).astype(int)
    fe_df['issue_y'] = df.issue_d.dt.year
    return fe_df


def features_eng(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    fe_df = _dates_fe(df, params)

    return fe_df
