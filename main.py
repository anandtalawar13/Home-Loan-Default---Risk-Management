"""
Home Loan Default Prediction API (FastAPI)

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

import os
import json
import warnings
from typing import Any, Dict, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# 1. Global Config & Warnings
# -------------------------------------------------------------------

warnings.filterwarnings("ignore")

# Folder where notebook saved models & metadata
SAVE_DIR = "saved_models"

if not os.path.isdir(SAVE_DIR):
    raise RuntimeError(
        f"Expected '{SAVE_DIR}' directory with models & metadata. "
        f"Current working directory: {os.getcwd()}"
    )


MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "logistic_regression": {
        "model_file": "logistic_regression",
        "threshold_key": "logistic_regression",
    },
    "random_forest": {
        "model_file": "random_forest",
        "threshold_key": "randomforest",
    },
    "lightgbm": {
        "model_file": "lightgbm",
        "threshold_key": "lightgbm",
    },
    "xgboost": {
        "model_file": "xgboost",
        "threshold_key": "xgboost",
    },
    "catboost": {
        "model_file": "catboost",
        "threshold_key": "catboost",
    },
    "easy_ensemble": {
        "model_file": "easy_ensemble",
        "threshold_key": "easyensemble",
    },
    "voting_ensemble": {
        "model_file": "voting_ensemble",
        "threshold_key": "ensemble_model",
    },
}


# -------------------------------------------------------------------
# 2. Utility Functions 
# -------------------------------------------------------------------

def load_artifacts(model_name: str):
    """
    Load model, feature columns list, and tuned threshold for deployment.
    Uses the same saved artifacts created in the notebook.
    """

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Allowed: {list(MODEL_REGISTRY.keys())}"
        )

    registry_entry = MODEL_REGISTRY[model_name]
    model_file_stem = registry_entry["model_file"]
    threshold_key = registry_entry["threshold_key"]

    # Load the model
    model_path = os.path.join(SAVE_DIR, f"{model_file_stem}.joblib")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Ensure you ran the notebook saving step."
        )
    model = joblib.load(model_path)

    # Load feature columns
    columns_path = os.path.join(SAVE_DIR, "columns.json")
    if not os.path.isfile(columns_path):
        raise FileNotFoundError(
            f"columns.json not found at {columns_path}. "
            f"Ensure you saved feature columns in the notebook."
        )
    with open(columns_path, "r") as f:
        model_columns = json.load(f)

    # Load thresholds
    thresholds_path = os.path.join(SAVE_DIR, "thresholds.json")
    if not os.path.isfile(thresholds_path):
        raise FileNotFoundError(
            f"thresholds.json not found at {thresholds_path}. "
            f"Ensure you saved best thresholds in the notebook."
        )
    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    if threshold_key not in thresholds:
        raise KeyError(
            f"Threshold key '{threshold_key}' not found in thresholds.json. "
            f"Available keys: {list(thresholds.keys())}"
        )

    threshold = thresholds[threshold_key]

    print(f"[INFO] Loaded model file: {model_file_stem}.joblib")
    print(f"[INFO] Using threshold key: {threshold_key} → {threshold}")
    print(f"[INFO] Loaded {len(model_columns)} feature columns.")

    return model, model_columns, threshold


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Column Cleaning — same method used during training.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    return df


def preprocess_input(df: pd.DataFrame, model_columns: List[str]) -> pd.DataFrame:
    """
    Clean column names + align missing columns + correct order.
    Same as notebook's 'preprocess_input'.
    """
    df = clean_cols(df.copy())

    # Add missing columns with zeros
    missing_cols = [col for col in model_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = 0

    # Remove unexpected columns
    df = df[model_columns]

    return df


def predict_default(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Load model, preprocess input, predict probability & label.

    Returns a DataFrame with:
        - default_probability
        - prediction (0/1)
    """
    # Load model + metadata
    model, model_columns, threshold = load_artifacts(model_name)

    # Preprocess incoming data
    df_prep = preprocess_input(df, model_columns)

    # Predict probability (class 1 = default)
    proba = model.predict_proba(df_prep)[:, 1]

    # Predict label using best threshold
    preds = (proba >= threshold).astype(int)

    # Build output
    output = pd.DataFrame(
        {
            "default_probability": proba,
            "prediction": preds,
        }
    )

    return output


def predict_single(customer_dict: Dict[str, Any], model_name: str) -> pd.DataFrame:
    """
    Accept a dictionary representing one customer.
    Exactly like the helper in the notebook.
    """
    df = pd.DataFrame([customer_dict])
    return predict_default(df, model_name)


# -------------------------------------------------------------------
# 3. Pydantic Schemas for FastAPI
# -------------------------------------------------------------------

AllowedModelName = Literal[
    "logistic_regression",
    "random_forest",
    "lightgbm",
    "xgboost",
    "catboost",
    "easy_ensemble",
    "voting_ensemble",
]


class SinglePredictionRequest(BaseModel):
    """
    Request schema for a single customer prediction.

    'features' should contain raw fields as used during training
    (e.g., AMT_INCOME_TOTAL, AMT_CREDIT, EXT_SOURCE_2, etc.).
    Missing fields will be filled with 0 during preprocessing.
    """
    model_name: AllowedModelName = Field(
        "catboost",
        description="Which model to use for prediction."
    )
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary of raw feature values for a single customer."
    )


class SinglePredictionResponse(BaseModel):
    model_name: str
    default_probability: float
    prediction: int


class BatchPredictionRequest(BaseModel):
    """
    Request schema for batch predictions.
    """
    model_name: AllowedModelName = Field(
        "catboost",
        description="Which model to use for prediction."
    )
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dictionaries. One dict per customer."
    )


class BatchPredictionItem(BaseModel):
    index: int
    default_probability: float
    prediction: int


class BatchPredictionResponse(BaseModel):
    model_name: str
    results: List[BatchPredictionItem]


# -------------------------------------------------------------------
# 4. FastAPI App Initialization
# -------------------------------------------------------------------

app = FastAPI(
    title="Home Loan Default Prediction API",
    description=(
        "FastAPI deployment of the Home Loan Default model.\n\n"
        "Models were trained on the Home Credit dataset with extensive "
        "feature engineering and threshold tuning. This API exposes:\n"
        "- Single-customer prediction endpoint\n"
        "- Batch prediction endpoint\n\n"
        "Use `/docs` for interactive Swagger UI."
    ),
    version="1.0.0",
)


# -------------------------------------------------------------------
# 5. API Routes
# -------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root() -> Dict[str, Any]:
    """
    Simple health check + metadata.
    """
    return {
        "status": "ok",
        "message": "Home Loan Default Prediction API is running 🚀",
        "available_models": list(MODEL_REGISTRY.keys()),
        "docs": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
        },
    }


@app.post("/predict-single", response_model=SinglePredictionResponse, tags=["Prediction"])
def predict_for_single(request: SinglePredictionRequest):
    """
    Predict default probability for a single customer.
    """
    try:
        result_df = predict_single(request.features, request.model_name)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    row = result_df.iloc[0]
    return SinglePredictionResponse(
        model_name=request.model_name,
        default_probability=float(row["default_probability"]),
        prediction=int(row["prediction"]),
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_for_batch(request: BatchPredictionRequest):
    """
    Predict default probabilities for multiple customers at once.
    """
    if len(request.records) == 0:
        raise HTTPException(status_code=400, detail="`records` list cannot be empty.")

    try:
        df = pd.DataFrame(request.records)
        result_df = predict_default(df, request.model_name)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    results: List[BatchPredictionItem] = []
    for idx, row in result_df.reset_index(drop=True).iterrows():
        results.append(
            BatchPredictionItem(
                index=idx,
                default_probability=float(row["default_probability"]),
                prediction=int(row["prediction"]),
            )
        )

    return BatchPredictionResponse(
        model_name=request.model_name,
        results=results,
    )


# -------------------------------------------------------------------
# 6. Local Dev Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
