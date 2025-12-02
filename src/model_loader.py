import json
import os
import joblib
from .config import MODEL_DIR, MODEL_REGISTRY

def load_artifacts(model_name: str):
    """
    Loads: 
    - trained model (.joblib)
    - feature columns (columns.json)
    - threshold (thresholds.json)
    """

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model_name: {model_name}")

    entry = MODEL_REGISTRY[model_name]
    model_file = entry["model_file"]
    threshold_key = entry["threshold_key"]

    # Load model
    model_path = os.path.join(MODEL_DIR, f"{model_file}.joblib")
    model = joblib.load(model_path)

    # Load feature columns
    columns_path = os.path.join(MODEL_DIR, "columns.json")
    with open(columns_path, "r") as f:
        columns = json.load(f)

    # Load threshold
    thresh_path = os.path.join(MODEL_DIR, "thresholds.json")
    with open(thresh_path, "r") as f:
        thresholds = json.load(f)

    threshold = thresholds[threshold_key]

    return model, columns, threshold
