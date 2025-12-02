# **Home Loan Default Prediction Report**

---

## **Introduction**

The objective of this project is to **predict the likelihood of a customer defaulting on a home loan** using demographic, financial, credit bureau, and repayment behavior data.

Accurate default prediction allows financial institutions to:
* Reduce **credit risk exposure**
* Improve **loan approval decisions**
* Optimize **interest rates based on risk**
* Enhance customer **risk profiling**
* Comply with regulatory risk requirements

This project uses a machine learning–driven approach and includes:
* Data preprocessing
* Imbalanced class handling
* Training advanced ML models
* Model comparison
* Threshold optimization
* Final production-ready deployment using FastAPI
* 
---

## **Dataset Overview**

The project uses multiple relational datasets:

| Dataset                 | Rows    | Purpose                        |
| ----------------------- | ------- | ------------------------------ |
| `application_train`     | 307,511 | Main training data with TARGET |
| `application_test`      | 48,744  | Prediction only                |
| `bureau`                | 1.7M+   | Past credit history            |
| `bureau_balance`        | 27M+    | Month-wise bureau status       |
| `previous_application`  | 1.6M+   | Previous loans                 |
| `installments_payments` | 13M+    | Repayment behavior             |
| `credit_card_balance`   | 3M+     | Credit card usage              |

**Target Variable**

| Value | Meaning                            |
| ----- | ---------------------------------- |
| 0     | Customer repaid loan               |
| 1     | **Customer defaulted (defaulter)** |

**Key Challenge: Imbalanced Data**
* Around **92% customers repay successfully**
* Only **8% are defaulters**

This imbalance makes it difficult for models to capture risky customers — requiring special techniques.

---

# **Challenges Faced & Techniques Used**

| Challenge                         | Problem Observed                            | Technique Used                           | Reason / Impact           |
| --------------------------------- | ------------------------------------------- | ---------------------------------------- | ------------------------- |
| Highly imbalanced target          | Model predicts all customers as non-default | Class Weights + Threshold Tuning         | Improves recall           |
| Dirty column names                | CatBoost, LightGBM could not train          | Universal column-cleaning function       | Ensures consistency       |
| Categorical features              | Not directly usable for ML                  | Label Encoding                           | Converts to numbers       |
| Outliers in financial variables   | Distorted model learning                    | IQR-based capping                        | Reduces skew              |
| Missing values                    | Missing financial & categorical data        | Median & frequency imputation            | Prevents information loss |
| Different feature sets train/test | CatBoost error: mismatched feature names    | Column alignment before training & infer | Prevents runtime failures |

---
# **Exploratory Data Analysis (EDA)**

**Target Distribution**
* Only ~8% are defaulters → **Very Imbalanced**
* Baseline accuracy is misleading (92% even with dumb model)
* Needed solutions:
  * Class weights
  * Threshold tuning
  * Ensemble models
---
**Missing Values**
* Few columns had **>30% missing data**
* Strategy:
  * Remove extremely missing columns (>40%)
  * Median imputation for numeric
  * Most frequent for categorical
---
**Outliers**
* Large outliers in:
  * Income
  * Credit amount
  * Employment duration

Used **Interquartile Range (IQR)** to cap extreme values.

---
**Categorical Analysis**
* Converted using **LabelEncoder**
* Features were later **one-hot encoded** wherever needed
---
**Correlation Insights**
* Low correlation between most features and target
* Linear models likely weak → **non-linear models (GBMs) required**
---

# **Data Preprocessing**

| Step                 | Description                                         |
| -------------------- | --------------------------------------------------- |
| Missing treatment    | Median / Mode Imputation                            |
| Outlier removal      | IQR capping                                         |
| Categorical encoding | LabelEncoder + One-hot encoding                     |
| Feature scaling      | Applied where needed (LogReg)                       |
| Column cleaning      | All features normalized and cleaned for consistency |
| Class imbalance      | **Class weights** instead of SMOTE                  |

---

## **Feature Engineering**
Important engineered features:
* Credit Utilization
* Debt-to-Income Ratio
* Late Payment Ratio
* Bureau Active/Closed Loans
* Max Overdue Amount
* Employment Stability Indicators

These features significantly improved model performance.

---

## **Model Performance Comparison**

| Model               | ROC-AUC    | PR-AUC     | Prec (Default) | Recall (Default) | F1 (Default) | Prec (Tuned) | Recall (Tuned) | F1 (Tuned) | Best Threshold |
| ------------------- | ---------- | ---------- | -------------- | ---------------- | ------------ | ------------ | -------------- | ---------- | -------------- |
| **CatBoost**        | **0.7799** | **0.2760** | 0.1829         | **0.6882**       | 0.2890       | **0.2835**   | 0.4109         | **0.3355** | 0.6852         |
| XGBoost             | 0.7777     | 0.2756     | 0.1914         | 0.6568           | **0.2964**   | 0.2596       | **0.4580**     | 0.3314     | 0.6349         |
| Ensemble            | 0.7789     | 0.2755     | **0.1916**     | 0.6582           | **0.2968**   | 0.2905       | 0.3823         | 0.3301     | 0.6313         |
| LightGBM            | 0.7772     | 0.2737     | 0.1841         | **0.6755**       | 0.2894       | 0.2618       | 0.4451         | 0.3297     | 0.6551         |
| EasyEnsemble        | 0.7589     | 0.2442     | 0.1674         | 0.6866           | 0.2692       | 0.2609       | 0.3728         | 0.3070     | 0.5723         |
| Random Forest       | 0.7571     | 0.2329     | **0.5926**     | 0.0032           | 0.0064       | 0.2525       | 0.3873         | 0.3057     | 0.1874         |
| Logistic Regression | 0.6697     | 0.1494     | 0.1298         | 0.6105           | 0.2141       | 0.1640       | 0.3778         | 0.2288     | 0.5837         |

### **Brief Observations**
* **CatBoost is the best overall model** with the highest ROC-AUC, PR-AUC, and tuned F1-score.
* **XGBoost provides the highest tuned recall**, making it excellent for identifying maximum defaulters.
* **LightGBM is the most stable and deployment-friendly**, with performance close to XGBoost and CatBoost.
* Logistic Regression and Random Forest underperform significantly in imbalanced settings.

**Final Choice:**
* **CatBoost** as the primary model.
* **XGBoost** when priority is maximum recall (catching more defaulters).
* **LightGBM** for fast, production-grade implementation.

---

# **Model Saving & Deployment**
```
saved_models/
│── logistic_regression.joblib
│── random_forest.joblib
│── lightgbm.joblib
│── xgboost.joblib
│── catboost.joblib
│── easy_ensemble.joblib
│── voting_ensemble.joblib
│── thresholds.json
└── columns.json
```

Where:
* **thresholds.json** = best thresholds from tuning
* **columns.json** = ensures correct feature alignment

---

## **Deployment Plan**
* The final model outputs **probability of default (PD)** for each customer.
* A tuned threshold (≈0.68 for CatBoost) is used to classify high-risk applicants.
* Deployment options:
  * API-based real-time scoring
  * Integration with loan approval systems
  * Batch scoring for portfolios

Monitoring includes quarterly retraining and drift detection.

---

## **Conclusion**

This project successfully:
* Processed large relational financial datasets
* Engineered meaningful behavioral and credit-risk features
* Evaluated multiple advanced ML models
* Identified **CatBoost** as the best-performing model
* Delivered insights for improving loan decision-making
---
















