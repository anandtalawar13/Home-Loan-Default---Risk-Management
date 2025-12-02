import re
import pandas as pd

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column names: removes special chars & normalizes names.
    Same logic as used during training.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    return df
