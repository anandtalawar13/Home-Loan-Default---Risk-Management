#  Home Loan Default Prediction

### End-to-End Machine Learning + FastAPI Deployment

<p align="center">

<!-- Python version -->

<img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />

<!-- FastAPI -->

<img src="https://img.shields.io/badge/FastAPI-Production%20API-009688?style=for-the-badge&logo=fastapi" /> 


---

#  Project Overview

This project builds a **production-ready Machine Learning system** that predicts whether a customer will default on a home loan.
It includes:

* Full data pipeline (cleaning → feature engineering → model training)
* Multiple ML models benchmarked
* Tuned thresholds for real-world credit scoring
* FastAPI backend for deployment
* Swagger API documentation
* Batch & single prediction support
* Saved artifacts for reproducibility

---

#  Features

* Advanced ML models (CatBoost, XGBoost, LightGBM, Ensembles)
* Production-grade FastAPI inference server
* Auto feature alignment & preprocessing pipeline
* Threshold-tuned predictions
* Batch processing support
* Clean, modular, and scalable structure

---

#  Model Performance Summary

| Model                       | ROC-AUC    | PR-AUC     | Best F1 (Tuned) |
| --------------------------- | ---------- | ---------- | --------------- |
| **CatBoost (Best Overall)** | **0.7799** | **0.2760** | **0.3355**      |
| XGBoost                     | 0.7777     | 0.2756     | 0.3314          |
| Ensemble Model              | 0.7789     | 0.2755     | 0.3301          |
| LightGBM                    | 0.7772     | 0.2737     | 0.3297          |
| Easy Ensemble               | 0.7589     | 0.2442     | 0.3070          |
| Random Forest               | 0.7571     | 0.2329     | 0.3057          |
| Logistic Regression         | 0.6697     | 0.1494     | 0.2288          |

###  Best Models

* **CatBoost:** Highest ROC-AUC, PR-AUC, F1-score
* **XGBoost:** Best recall (most defaulters caught)

---

#  Architecture Overview

* Raw data → Cleaning
* Feature engineering (multi-table)
* Model training & evaluation
* Threshold tuning
* Model saving (joblib + JSON metadata)
* FastAPI inference server
* Swagger documentation
* Batch & single prediction endpoints
---

#  FastAPI Deployment

###  Start API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

###  API Documentation

* Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc → [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

#  Example Requests

### Single Prediction

```json
{
  "model_name": "catboost",
  "features": {
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 450000,
    "EXT_SOURCE_2": 0.41,
    "DAYS_EMPLOYED": -2000,
    "NAME_FAMILY_STATUS": "Married"
  }
}
```

### Example Response

```json
{
  "model_name": "catboost",
  "default_probability": 0.28412,
  "prediction": 0
}
```

---

#  Key ML Features

* Multi-table feature engineering
* Missing value handling & anomaly correction
* Outlier capping
* Threshold optimization for F1/Recall
* Model versioning with metadata
* Batch inference for portfolios

---

#  Installation

```bash
pip install -r requirements.txt
```

---

#  Support

If you like this project, please **⭐ star the repository**!

---


Just tell me!
