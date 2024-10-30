"""
This is a boilerplate pipeline 'encode'
generated using Kedro 0.19.9
"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Function to encode emp_length to number
def _parse_emp_len(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df[params['emp_len']] = df[params['emp_len']].str.split(" ").str[0].str.replace("+", "").str.replace("<", "0").astype(int)
    return df

# Function to encode default_status from loan_status
def _default_status(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df['default_status'] = (
            (df[params['default']] == 'Charged Off') |
            (df[params['default']] == 'Does not meet the credit policy. Status:Charged Off') |
            (df[params['default']] == 'Default')
    )
    return df.drop(params['default'], axis=1)

# Function encodes object values to numbers
def _encode(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    encoder = OrdinalEncoder()
    df[params['category']] = encoder.fit_transform(df[params['category']])
    return df

def encode_dataset(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = _parse_emp_len(df, params)
    df = _default_status(df, params)
    df = _encode(df, params)
    return df
