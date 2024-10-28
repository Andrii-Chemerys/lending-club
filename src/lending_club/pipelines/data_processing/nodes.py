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
    return x.where(x.isna(), x=="Y")
