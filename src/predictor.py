import pandas as pd
from .model_loader import load_artifacts
from .preprocessing import preprocess_input

def predict_default(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Makes prediction using selected ML model.
    Returns:
    - default_probability
    - prediction (0/1)
    """

    model, model_columns, threshold = load_artifacts(model_name)

    df_processed = preprocess_input(df, model_columns)
    probabilities = model.predict_proba(df_processed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame({
        "default_probability": probabilities,
        "prediction": predictions
    })

def predict_single(data: dict, model_name: str) -> pd.DataFrame:
    df = pd.DataFrame([data])
    return predict_default(df, model_name)
