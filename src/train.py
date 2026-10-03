"""
Training & Evaluation Pipeline for Banking Fraud Detection.
Implements:
- Leak-free Stratified 80/20 train-test split
- Train-only outlier capping and robust scaling
- 5-Fold Stratified Cross-Validation on training data
- Model training: Logistic Regression (balanced), Random Forest (balanced), XGBoost (scale_pos_weight)
- Validation-based threshold tuning (Holdout test set is kept untouched until final evaluation)
- Diagnostic curve generation and artifact persistence
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocessing import create_preprocessor, OutlierCapper
from src.evaluate import compute_metrics, find_optimal_thresholds, generate_diagnostic_curves


def main():
    print("=" * 60)
    print("SENTINEL FRAUD DETECTION PIPELINE: TRAINING & EVALUATION")
    print("=" * 60)

    # 1. Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "creditcard.csv")
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]:,} records, {df.shape[1]} columns.")
    print(f"Class breakdown: Legitimate = {(df['Class']==0).sum():,} | Fraud = {(df['Class']==1).sum():,}")

    # 2. Stratified Train/Test Split (80/20)
    features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    X = df[features]
    y = df['Class']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split: {X_train_raw.shape[0]:,} samples (Fraud: {(y_train==1).sum():,})")
    print(f"Test split:  {X_test_raw.shape[0]:,} samples (Fraud: {(y_test==1).sum():,})")

    # 3. Fit Preprocessing strictly on Training Split
    print("\nFitting Preprocessor (OutlierCapper + RobustScaler) on Train split only...")
    from src.preprocessing import OutlierCapper
    from sklearn.preprocessing import RobustScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    pca_features = [f"V{i}" for i in range(1, 29)]
    all_features = ['Time'] + pca_features + ['Amount']
    
    preprocessor = Pipeline([
        ('capper', OutlierCapper(lower_quantile=0.001, upper_quantile=0.999, columns=all_features)),
        ('scaler', ColumnTransformer(
            transformers=[
                ('robust_amount', RobustScaler(), ['Amount']),
                ('robust_time', RobustScaler(), ['Time']),
                ('passthrough_pca', 'passthrough', pca_features)
            ],
            remainder='passthrough'
        ))
    ])

    preprocessor.fit(X_train_raw)
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))

    # Transform features
    transformed_cols = ['Amount', 'Time'] + pca_features
    X_train_trans_arr = preprocessor.transform(X_train_raw)
    X_test_trans_arr = preprocessor.transform(X_test_raw)

    X_train_trans = pd.DataFrame(X_train_trans_arr, columns=transformed_cols, index=X_train_raw.index)
    X_test_trans = pd.DataFrame(X_test_trans_arr, columns=transformed_cols, index=X_test_raw.index)

    # 4. Model Definitions
    imbalance_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Imbalance ratio in training data: {imbalance_ratio:.2f} : 1")

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=imbalance_ratio,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
    }

    # 5. 5-Fold Stratified Cross-Validation on Training Data
    print("\n--- Running 5-Fold Stratified Cross-Validation on Training Data ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {}
    oof_predictions = {}

    for name, model in models.items():
        print(f"Cross-validating: {name}...")
        oof_probs = np.zeros(len(X_train_trans))
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_trans, y_train)):
            X_fold_train = X_train_trans.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train_trans.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Clone and fit
            m = model.__class__(**model.get_params())
            m.fit(X_fold_train, y_fold_train)
            
            val_probs = m.predict_proba(X_fold_val)[:, 1]
            oof_probs[val_idx] = val_probs
            
            m_metrics = compute_metrics(y_fold_val, val_probs, threshold=0.5)
            fold_metrics.append(m_metrics)

        oof_predictions[name] = oof_probs
        
        # Aggregate CV metrics
        cv_pr_auc = float(np.mean([fm['pr_auc'] for fm in fold_metrics]))
        cv_roc_auc = float(np.mean([fm['roc_auc'] for fm in fold_metrics]))
        cv_f1 = float(np.mean([fm['f1'] for fm in fold_metrics]))
        cv_prec = float(np.mean([fm['precision'] for fm in fold_metrics]))
        cv_rec = float(np.mean([fm['recall'] for fm in fold_metrics]))

        cv_results[name] = {
            "cv_pr_auc": cv_pr_auc,
            "cv_roc_auc": cv_roc_auc,
            "cv_f1": cv_f1,
            "cv_precision": cv_prec,
            "cv_recall": cv_rec
        }
        print(f"  [{name}] CV PR-AUC: {cv_pr_auc:.4f} | CV ROC-AUC: {cv_roc_auc:.4f} | CV F1: {cv_f1:.4f} (at 0.5)")

    # 6. Threshold Calibration using Out-Of-Fold Training Validation Predictions
    print("\n--- Calibrating Decision Thresholds on Validation Out-Of-Fold Probabilities ---")
    threshold_data = {}
    for name in models.keys():
        opt_thresh = find_optimal_thresholds(y_train, oof_predictions[name])
        threshold_data[name] = opt_thresh
        print(f"  [{name}] Calibrated Best F1 Threshold: {opt_thresh['best_f1_threshold']:.2f} (CV F1: {opt_thresh['best_f1_score']:.4f})")

    # 7. Final Training on Full Training Set & Evaluation on Holdout Test Set
    print("\n--- Final Model Training on X_train & Evaluation on Untouched Test Split ---")
    final_test_metrics = {}
    diagnostic_curves = {}
    saved_models = {}

    for name, model in models.items():
        print(f"Fitting final {name} on full training set...")
        model.fit(X_train_trans, y_train)
        
        # Save model
        clean_name = name.lower().replace(" ", "_")
        model_file = os.path.join(models_dir, f"{clean_name}.joblib")
        joblib.dump(model, model_file)
        saved_models[name] = model_file

        # Predict on holdout test set
        test_probs = model.predict_proba(X_test_trans)[:, 1]
        
        # Metrics at default 0.5 threshold
        default_metrics = compute_metrics(y_test, test_probs, threshold=0.5)
        
        # Metrics at CV-calibrated threshold
        calib_thresh = threshold_data[name]['best_f1_threshold']
        calibrated_metrics = compute_metrics(y_test, test_probs, threshold=calib_thresh)
        
        # Curves
        curves = generate_diagnostic_curves(y_test, test_probs)
        diagnostic_curves[name] = curves

        final_test_metrics[name] = {
            "default_threshold_0.5": default_metrics,
            "calibrated_threshold": calibrated_metrics,
            "calibrated_threshold_value": calib_thresh,
            "cv_metrics": cv_results[name]
        }

        print(f"  [{name}] TEST PR-AUC: {default_metrics['pr_auc']:.4f} | ROC-AUC: {default_metrics['roc_auc']:.4f}")
        print(f"  [{name}] Default (0.50): Prec={default_metrics['precision']:.4f}, Rec={default_metrics['recall']:.4f}, F1={default_metrics['f1']:.4f}")
        print(f"  [{name}] Calibrated ({calib_thresh:.2f}): Prec={calibrated_metrics['precision']:.4f}, Rec={calibrated_metrics['recall']:.4f}, F1={calibrated_metrics['f1']:.4f}")

    # 8. Create sample test evaluation dataset for lightning-fast UI demonstration
    # Pick XGBoost as champion for primary predictions, include true labels and model probabilities
    primary_model = models["XGBoost"]
    primary_test_probs = primary_model.predict_proba(X_test_trans)[:, 1]
    
    test_eval_df = X_test_raw.copy()
    test_eval_df['Ground_Truth'] = y_test.values
    test_eval_df['Fraud_Probability'] = np.round(primary_test_probs * 100, 2)
    test_eval_df['Fraud_Prob_Raw'] = primary_test_probs
    
    # Save test sample dataset (all fraud cases from test set + representative legitimate cases)
    fraud_cases = test_eval_df[test_eval_df['Ground_Truth'] == 1]
    legit_sample = test_eval_df[test_eval_df['Ground_Truth'] == 0].sample(n=min(2000, len(test_eval_df[test_eval_df['Ground_Truth'] == 0])), random_state=42)
    compact_test_demo = pd.concat([fraud_cases, legit_sample]).sort_index()
    compact_test_demo.to_csv(os.path.join(outputs_dir, "test_demo_pool.csv"), index=True)

    # 9. Save all artifacts
    with open(os.path.join(outputs_dir, "model_metrics.json"), "w") as f:
        json.dump(final_test_metrics, f, indent=2)

    with open(os.path.join(outputs_dir, "threshold_calibration.json"), "w") as f:
        json.dump(threshold_data, f, indent=2)

    with open(os.path.join(outputs_dir, "diagnostic_curves.json"), "w") as f:
        json.dump(diagnostic_curves, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION COMPLETE")
    print(f"Saved artifacts to:\n  - Models: {models_dir}\n  - Outputs: {outputs_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
