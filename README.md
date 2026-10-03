# SENTINEL: AI-Powered Transaction Risk Intelligence Platform

> **College Case-Study Project:** *Banking Fraud Detection Pipeline with Outlier Capping & Precision-Recall Diagnostics*  
> **Tagline:** *Detect. Investigate. Prevent.*

---

## 🛡️ Project Overview

**SENTINEL** is an end-to-end Machine Learning fraud detection and risk intelligence platform built for financial transaction streams. Built upon the ULB Credit Card Fraud dataset (284,807 transactions with an extreme 578:1 class imbalance), the system addresses the critical challenges of real-world banking fraud detection:

1. **Heavy-Tailed Distributions & Outliers:** Extreme amounts and PCA signal variance without dropping true fraud records.
2. **Extreme Class Imbalance (0.173% Fraud Rate):** Accuracy is a misleading vanity metric; models are optimized for Precision-Recall Area Under Curve (PR-AUC) and operational threshold calibration.
3. **Zero Data Leakage:** Preprocessing (Quantile Outlier Capper & Robust Scaling) is fitted strictly on training splits.
4. **Validation-Based Threshold Tuning:** Decision boundaries are calibrated on 5-fold cross-validation out-of-fold predictions, keeping the holdout test set completely untouched until final evaluation.
5. **Explainability & Triage:** Mathematical SHAP (Shapley Additive Explanations) feature contribution signals provide transparent transaction dossiers for fraud analysts.

---

## 🏗️ Architecture & Project Structure

```
.
├── creditcard.csv                 # Original raw dataset (untouched)
├── models/                        # Serialized ML models and preprocessors
│   ├── preprocessor.joblib        # OutlierCapper + RobustScaler pipeline (train-fitted)
│   ├── xgboost.joblib             # Champion Gradient Boosted Classifier
│   ├── random_forest.joblib       # Balanced Random Forest Ensemble
│   └── logistic_regression.joblib # Balanced Logistic Regression Baseline
├── outputs/                       # Real evaluation artifacts & diagnostic curves
│   ├── model_metrics.json         # Real PR-AUC, ROC-AUC, Precision, Recall, F1
│   ├── threshold_calibration.json # Cross-validation threshold sweeps
│   ├── diagnostic_curves.json     # Sampled coordinates for PR and ROC curves
│   └── test_demo_pool.csv         # Holdout test set sample pool with real probabilities
├── src/                           # Core ML Engine
│   ├── preprocessing.py           # OutlierCapper & RobustScaler pipelines
│   ├── train.py                   # 5-Fold Stratified CV, training, & calibration
│   ├── evaluate.py                # Diagnostic curves, confusion matrices, metrics
│   ├── predict.py                 # Real-time & batch inference engine
│   └── explain.py                 # SHAP TreeExplainer & Feature Contribution module
├── app/                           # Fintech Streamlit Web Application
│   ├── app.py                     # 5-view interactive dashboard
│   ├── styles.py                  # Dark cybersecurity CSS theme
│   └── components.py              # Reusable gauges, KPI cards, charts
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 📊 Model Benchmark & Real Metrics (Holdout Test Set)

All metrics are derived from the holdout test set (56,962 transactions, 98 fraud cases):

| Model | Test PR-AUC | Test ROC-AUC | Calibrated Threshold | Test Precision | Test Recall | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **0.8610** | **0.9800** | **0.90** | **0.8182** | **0.8265** | **0.8223** |
| **Random Forest** | 0.8513 | 0.9751 | 0.52 | 0.8280 | 0.7857 | 0.8063 |
| **Logistic Regression** | 0.7757 | 0.9729 | 0.99 | 0.5608 | 0.8469 | 0.6748 |

> **Key Takeaway on Threshold Calibration:**  
> When handling extreme class imbalance with cost-sensitive weighting, default 0.50 thresholds produce excessive false alarms for weighted linear/boosted models. Calibrating the threshold on cross-validation data boosts XGBoost Precision from **64.6% → 81.8%** and Logistic Regression Precision from **5.7% → 56.1%** while maintaining over 82–84% recall.

---

## 🚀 Running the Prototype Locally

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-train & Evaluate ML Pipeline
```bash
python -m src.train
```

### 3. Launch Sentinel Prototype
```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 🎓 3–5 Minute Faculty Demo Guide

1. **Overview:** Show dataset volume (284,807 tx), extreme 0.173% fraud rate, and diurnal 24-hour transaction volume patterns.
2. **Transaction Scanner:** Select a verified transaction from the holdout pool, click **SCAN TRANSACTION**, view the dynamic cyber telemetry check and Plotly risk score gauge.
3. **Fraud Investigation:** Open the case dossier to reveal mathematical SHAP feature contributions showing which PCA signals triggered the fraud alert.
4. **Model Intelligence:** Demonstrate the PR curve and the **Interactive Threshold Simulator** — adjust the threshold slider from 0.10 to 0.90 to show the live trade-off between Precision and Recall.
