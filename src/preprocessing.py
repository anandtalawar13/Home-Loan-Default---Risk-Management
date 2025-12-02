import pandas as pd
from .utils import clean_cols

def preprocess_input(df: pd.DataFrame, model_columns: list) -> pd.DataFrame:
    """
    Prepares incoming data for ML prediction:
    - Clean column names
    - Add missing columns (set to 0)
    - Order columns correctly
    """
    df = clean_cols(df.copy())

    # Add missing columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only required columns, in correct order
    df = df[model_columns]

    return df
