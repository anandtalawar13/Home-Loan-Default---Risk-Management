# Home Loan Default Prediction

### **End-to-End Machine Learning + FastAPI Deployment**

<p align="center">
  <img src="https://img.icons8.com/?size=200&id=59914&format=png&color=4CAF50" width="140" />
</p>

<p align="center">
  <b>Smart Credit Risk Analysis | ML Models | FastAPI | Threshold Tuning</b>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/main/ML/machine-learning.gif" width="500"/>
</p>

---

# Project Overview

This project builds a **production-ready Machine Learning system** that predicts whether a customer will **default on a home loan**.
It includes:

* Full data pipeline (cleaning → feature engineering → model training)
* Multiple ML models benchmarked
* Tuned thresholds for real-world credit scoring
* **FastAPI** backend for deployment
* Ready-to-use Swagger documentation
* Batch & Single predictions
* Saved artifacts for reproducibility

---

# Features

<p align="center">
  <img src="https://i.imgur.com/7yUVEeT.gif" width="450"/>
</p>

* Advanced ML models (CatBoost, XGBoost, LightGBM, Ensembles)
* Production FastAPI server
* Auto feature alignment
* Threshold-tuned predictions
* Batch processing
* Reproducible model artifacts
* Clean & scalable code structure

---

# Model Performance Summary

| Model                       | ROC-AUC    | PR-AUC     | Best F1 (Tuned) |
| --------------------------- | ---------- | ---------- | --------------- |
| **CatBoost (Best Overall)** | **0.7799** | **0.2760** | **0.3355**      |
| XGBoost                     | 0.7777     | 0.2756     | 0.3314          |
| Ensemble Model              | 0.7789     | 0.2755     | 0.3301          |
| LightGBM                    | 0.7772     | 0.2737     | 0.3297          |
| Easy Ensemble               | 0.7589     | 0.2442     | 0.3070          |
| Random Forest               | 0.7571     | 0.2329     | 0.3057          |
| Logistic Regression         | 0.6697     | 0.1494     | 0.2288          |

### Best Model

**CatBoost** → Highest ROC-AUC, PR-AUC, F1-score
**XGBoost** → Best recall (catches most defaulters)

---

# Architecture Diagram

<p align="center">
  <img src="https://i.imgur.com/2s6JajB.png" width="800"/>
</p>

---

# Repository Structure

```
├── notebooks/
│   └── home_loan_default.ipynb
├── saved_models/
│   ├── catboost.joblib
│   ├── xgboost.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── lightgbm.joblib
│   ├── easy_ensemble.joblib
│   ├── voting_ensemble.joblib
│   ├── columns.json
│   └── thresholds.json
├── main.py                    # FastAPI service
├── requirements.txt
└── README.md
```

---

# FastAPI Deployment

<p align="center">
  <img src="https://i.imgur.com/2Au7M3e.gif" width="500"/>
</p>

### Start API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 📘 API Documentation

* **Swagger UI** → [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc** → [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

# Example Requests

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

# Key ML Features

<p align="center">
  <img src="https://i.imgur.com/TcIHZVn.gif" width="450"/>
</p>

* Multi-table feature engineering
* Handling missing values & anomalies
* Outlier capping and transformation
* Threshold tuning for recall/F1 optimization
* Model versioning and metadata storage
* Batch scoring support

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Future Enhancements

* SHAP explainability
* Docker container deployment
* Streamlit dashboard (credit scoring UI)
* AWS/GCP cloud deployment
* Automated retraining pipeline (CI/CD)

---

# Contributing

Contributions are welcome – open a PR or issue any time!

<p align="center">
  <img src="https://raw.githubusercontent.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/main/Hand%20Gifs/handshake.gif" width="200"/>
</p>

---


# Enjoyed this Project?

**Star this repository** to support continuous development!

<p align="center">
  <img src="https://i.imgur.com/Q18F5kL.gif" width="300"/>
</p>

---

