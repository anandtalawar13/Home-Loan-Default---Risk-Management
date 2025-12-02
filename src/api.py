from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd

from .predictor import predict_single, predict_default
from .config import MODEL_REGISTRY


app = FastAPI(
    title="Home Loan Default Prediction API",
    description="FastAPI ML service for predicting home loan default risk.",
    version="1.0.0"
)


class SinglePredictionRequest(BaseModel):
    model_name: str
    features: Dict


class BatchPredictionRequest(BaseModel):
    model_name: str
    records: List[Dict]


@app.get("/")
def root():
    return {
        "status": "running",
        "available_models": list(MODEL_REGISTRY.keys()),
        "endpoints": {
            "single": "/predict-single",
            "batch": "/predict-batch"
        }
    }


@app.post("/predict-single")
def single_prediction(request: SinglePredictionRequest):
    try:
        df = predict_single(request.features, request.model_name)
        result = df.iloc[0]
        return {
            "model_name": request.model_name,
            "default_probability": float(result["default_probability"]),
            "prediction": int(result["prediction"])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-batch")
def batch_prediction(request: BatchPredictionRequest):
    try:
        df_input = pd.DataFrame(request.records)
        df_out = predict_default(df_input, request.model_name)

        return {
            "model_name": request.model_name,
            "results": df_out.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
