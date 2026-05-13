import pandas as pd

from src.config import DATA_PATH


def load_dataset():
    """
    Load and clean machining dataset.
    """

    df = pd.read_csv(DATA_PATH)

    # Remove whitespace from column names
    df = df.rename(columns=lambda x: x.strip())

    # Fix column naming inconsistency
    if "Ax" in df.columns and "ap" not in df.columns:
        df = df.rename(columns={"Ax": "ap"})

    # Keep required columns only
    required_columns = ["Vc", "Fz", "ap", "Ra"]

    df = df[required_columns]

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove invalid rows
    df = df.dropna().reset_index(drop=True)

    return df
