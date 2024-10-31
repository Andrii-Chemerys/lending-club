"""
This is a boilerplate pipeline 'feature_eng'
generated using Kedro 0.19.9
"""
import pandas as pd

# Define dates feature engineering function
def _dates_fe(df: pd.DataFrame, fe_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    # Calculate how many months ago the earliest credit line was open
    fe_df['mo_since_earliest_cr_line'] = (
        (pd.Timestamp(params['cur_date']) - df.earliest_cr_line) / pd.Timedelta(30, 'D')
    ).astype(int)
    fe_df['issue_y'] = df.issue_d.dt.year
    return fe_df

# Considering joint applications let's use new 'adjusted' features for annual income, dti and revol_bal
# that takes joint features where it is the case or individual features otherwise
def _adjusted_feat(df: pd.DataFrame, fe_df: pd.DataFrame) -> pd.DataFrame:
    fe_df['dti_adj'] = df.dti_joint.where(df.application_type == "Joint App", df.dti)
    fe_df['revol_bal_adj'] = df.revol_bal_joint.where(df.application_type == "Joint App", df.revol_bal)
    fe_df['annual_inc_adj'] = df.annual_inc_joint.where(df.application_type == "Joint App", df.annual_inc)

    # List of features of second applicant and aggregation function
    # that will be used to produce adjusted feature
    sec_extra_features=[
        ['sec_app_chargeoff_within_12_mths', 'sum'],
        ['sec_app_fico_range_high','max'],
        ['sec_app_inq_last_6mths', 'sum'],
        ['sec_app_mort_acc', 'sum'],
        ['sec_app_num_rev_accts','sum'],
        ['sec_app_open_acc','sum'],
        ['sec_app_open_act_il','sum'],
        ['sec_app_revol_util','avg']
        ]
    # Iterate trough sec_extra_feature list to make new features and remove original
    # features from model's features list
    for sec_feat, func in sec_extra_features:
        feat = sec_feat[8:]                                     # individual's feature name
        feat_adj = feat + "_adj"                                # new feature name
        if func == 'max':
            fe_df[feat_adj] = df[[feat, sec_feat]].max(axis=1)
        elif func == 'sum':
            fe_df[feat_adj] = df[[feat, sec_feat]].sum(axis=1)
        elif func == 'avg':
            fe_df[feat_adj] = df[[feat, sec_feat]].mean(axis=1)
    return fe_df

def features_eng(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    # Feature engineering dataset will be new dataframe
    fe_df = pd.DataFrame()
    fe_df = _dates_fe(df, fe_df, params)
    fe_df = _adjusted_feat(df, fe_df)
    return fe_df