"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.9
"""

import pandas as pd

# Define functions to parse different kind of values to numbers
def _parse_pct(x: pd.Series) -> pd.Series:
    return x.str.replace("%", "").astype(float)

def _parse_term(x: pd.Series) -> pd.Series:
    return x.str.replace(" months", "").astype(int)

def _parse_date(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, format="%b-%Y")

def _parse_bool(x: pd.Series) -> pd.Series:
    return x=="Y"

# Define function that process dataset with
def processing_lc(df: pd.DataFrame) -> pd.DataFrame:
    # Process term values ('XX months' to XX)
    df.term = _parse_term(df.term)
    # Process percentage values
    pct_list = ['int_rate', 'revol_util']
    for x in pct_list:
        df[x] = _parse_pct(df[x])
    # Process date values
    date_list = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d',
                'sec_app_earliest_cr_line', 'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date']
    for x in date_list:
        df[x] = _parse_date(df[x])
    # Process boolean values
    bool_list = ['hardship_flag', 'debt_settlement_flag']
    for x in bool_list:
        df[x] = _parse_bool(df[x])
    df.set_index('id', drop=True, inplace=True)
    return df
