"""
Prediction & Risk Scoring Module for Banking Fraud Detection Pipeline.
Loads trained models, runs leak-free inference, categorizes transaction risk,
and generates analyst recommendation signals.
"""

import os
import joblib
import numpy as np
import pandas as pd

EXPECTED_FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']


def get_risk_category(prob, threshold=0.5):
    """
    Categorizes risk based on continuous fraud probability and tuned decision threshold.
    """
    if prob >= threshold:
        return "HIGH RISK", "Urgent Investigation & Hold"
    elif prob >= max(0.15, threshold * 0.4):
        return "MEDIUM RISK", "Secondary Verification Required"
    else:
        return "LOW RISK", "Approve Transaction"


class FraudPredictor:
    """
    Loads saved model pipeline and serves real-time or batch inference.
    """
    def __init__(self, model_path, preprocessor_path=None, threshold=0.5):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path) if preprocessor_path and os.path.exists(preprocessor_path) else None
        self.threshold = threshold

    def validate_dataframe(self, df):
        """
        Validates structure and data types of input transaction DataFrame.
        """
        missing_cols = [col for col in EXPECTED_FEATURES if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Check non-numeric values
        for col in EXPECTED_FEATURES:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Attempt conversion
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    return False, f"Column '{col}' contains invalid non-numeric values."
        
        # Check for NaN / null values
        if df[EXPECTED_FEATURES].isnull().any().any():
            return False, "Dataset contains null or NaN values in feature columns."
            
        return True, "Validation successful."

    def predict_instance(self, row_dict_or_series, threshold=None):
        """
        Runs prediction on a single transaction dictionary or Series.
        """
        t = threshold if threshold is not None else self.threshold
        if isinstance(row_dict_or_series, dict):
            df_single = pd.DataFrame([row_dict_or_series])
        elif isinstance(row_dict_or_series, pd.Series):
            df_single = pd.DataFrame([row_dict_or_series.to_dict()])
        else:
            df_single = pd.DataFrame(row_dict_or_series)
            
        valid, msg = self.validate_dataframe(df_single)
        if not valid:
            raise ValueError(msg)
            
        X = df_single[EXPECTED_FEATURES]
        if self.preprocessor is not None:
            X_trans = self.preprocessor.transform(X)
            # Re-form DataFrame with correct columns if needed
            if not isinstance(X_trans, pd.DataFrame):
                X_trans = pd.DataFrame(X_trans, columns=['Amount', 'Time'] + [f'V{i}' for i in range(1, 29)])
            probs = self.model.predict_proba(X_trans)[:, 1]
        else:
            probs = self.model.predict_proba(X)[:, 1]
            
        prob = float(probs[0])
        pred = 1 if prob >= t else 0
        risk_label, action = get_risk_category(prob, t)
        
        return {
            "fraud_probability": prob,
            "prediction": pred,
            "prediction_label": "FRAUD" if pred == 1 else "LEGITIMATE",
            "threshold": t,
            "risk_level": risk_label,
            "recommended_action": action,
            "features_df": X
        }

    def predict_batch(self, df, threshold=None):
        """
        Runs batch predictions on a DataFrame of transactions.
        """
        t = threshold if threshold is not None else self.threshold
        valid, msg = self.validate_dataframe(df)
        if not valid:
            raise ValueError(msg)
            
        X = df[EXPECTED_FEATURES]
        if self.preprocessor is not None:
            X_trans = self.preprocessor.transform(X)
            if not isinstance(X_trans, pd.DataFrame):
                X_trans = pd.DataFrame(X_trans, columns=['Amount', 'Time'] + [f'V{i}' for i in range(1, 29)])
            probs = self.model.predict_proba(X_trans)[:, 1]
        else:
            probs = self.model.predict_proba(X)[:, 1]
            
        preds = (probs >= t).astype(int)
        
        results_df = df.copy()
        results_df['Fraud_Probability'] = np.round(probs * 100, 2)
        results_df['Prediction'] = np.where(preds == 1, 'FRAUD', 'LEGITIMATE')
        
        risk_levels = []
        actions = []
        for p in probs:
            r_lvl, act = get_risk_category(p, t)
            risk_levels.append(r_lvl)
            actions.append(act)
            
        results_df['Risk_Level'] = risk_levels
        results_df['Action'] = actions
        
        return results_df
