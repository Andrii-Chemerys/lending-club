"""
This is a boilerplate pipeline 'encode'
generated using Kedro 0.19.9
"""
import pandas as pd

def _parse_emp_len(x: pd.Series) -> pd.Series:
    return x.str.split(" ").str[0].str.replace("+", "").str.replace("<", "0").astype(int)