"""Input validation and safeguard checks."""

import pandas as pd

def validate_input(df: pd.DataFrame):
    if df.isnull().any().any():
        raise ValueError("Missing values detected")

    if (df < 0).any().any():
        raise ValueError("Negative values not allowed")
