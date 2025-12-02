import os

# Directory where models and metadata (columns.json & thresholds.json) are saved
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# Registry maps public model names to filenames + threshold keys
MODEL_REGISTRY = {
    "logistic_regression": {"model_file": "logistic_regression", "threshold_key": "logistic_regression"},
    "random_forest": {"model_file": "random_forest", "threshold_key": "randomforest"},
    "lightgbm": {"model_file": "lightgbm", "threshold_key": "lightgbm"},
    "xgboost": {"model_file": "xgboost", "threshold_key": "xgboost"},
    "catboost": {"model_file": "catboost", "threshold_key": "catboost"},
    "easy_ensemble": {"model_file": "easy_ensemble", "threshold_key": "easyensemble"},
    "voting_ensemble": {"model_file": "voting_ensemble", "threshold_key": "ensemble_model"},
}
