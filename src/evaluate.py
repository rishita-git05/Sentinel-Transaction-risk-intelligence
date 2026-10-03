"""
Evaluation and Diagnostic Module for Fraud Detection Pipeline.
Calculates PR-AUC, ROC-AUC, precision-recall trade-offs, confusion matrices,
and threshold sweeps strictly on validation splits or holdout evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes comprehensive classification metrics at a given decision threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    return {
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "accuracy": float(acc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_fraud_detected": int(tp + fp)
    }


def find_optimal_thresholds(y_val, y_val_prob):
    """
    Sweeps thresholds on validation / CV probability predictions.
    Identifies:
    1. Best F1 threshold
    2. High-Recall threshold (Recall >= 0.85)
    3. Balanced Precision-Recall threshold
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    records = []
    for t in thresholds:
        m = compute_metrics(y_val, y_val_prob, threshold=t)
        records.append(m)
    
    df_thresh = pd.DataFrame(records)
    
    # Best F1
    best_f1_idx = df_thresh['f1'].idxmax()
    best_f1_thresh = df_thresh.loc[best_f1_idx, 'threshold']
    
    # High recall >= 0.85 with maximum precision
    high_rec_candidates = df_thresh[df_thresh['recall'] >= 0.85]
    if not high_rec_candidates.empty:
        high_rec_idx = high_rec_candidates['precision'].idxmax()
        high_rec_thresh = high_rec_candidates.loc[high_rec_idx, 'threshold']
    else:
        high_rec_thresh = 0.20
        
    return {
        "threshold_sweep": records,
        "best_f1_threshold": float(best_f1_thresh),
        "best_f1_score": float(df_thresh.loc[best_f1_idx, 'f1']),
        "high_recall_threshold": float(high_rec_thresh),
        "default_threshold": 0.50
    }


def generate_diagnostic_curves(y_true, y_prob, n_points=200):
    """
    Computes sampled PR and ROC curve coordinate data for lightweight UI rendering.
    """
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    # Downsample points for fast plotting
    if len(precision) > n_points:
        idx = np.linspace(0, len(precision) - 1, n_points, dtype=int)
        pr_data = {
            "precision": precision[idx].tolist(),
            "recall": recall[idx].tolist(),
            "thresholds": pr_thresholds[np.clip(idx[:-1], 0, len(pr_thresholds)-1)].tolist() + [1.0]
        }
    else:
        pr_data = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist() + [1.0]
        }
        
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    if len(fpr) > n_points:
        idx_roc = np.linspace(0, len(fpr) - 1, n_points, dtype=int)
        roc_data = {
            "fpr": fpr[idx_roc].tolist(),
            "tpr": tpr[idx_roc].tolist(),
            "thresholds": roc_thresholds[idx_roc].tolist()
        }
    else:
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist()
        }
        
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    
    return {
        "pr_curve": pr_data,
        "roc_curve": roc_data,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    }
