"""
Preprocessing module for Banking Fraud Detection Pipeline.
Ensures zero data leakage: transformers are fit strictly on training splits.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Clips extreme numerical values based on quantiles learned strictly during fit.
    Prevents outlier explosion without discarding minority fraud instances.
    """
    def __init__(self, lower_quantile=0.001, upper_quantile=0.999, columns=None):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.columns = columns
        self.bounds_ = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            cols = self.columns if self.columns is not None else X.columns
            for col in cols:
                q_low = X[col].quantile(self.lower_quantile)
                q_high = X[col].quantile(self.upper_quantile)
                self.bounds_[col] = (q_low, q_high)
        else:
            X_arr = np.asarray(X)
            cols_idx = range(X_arr.shape[1])
            for idx in cols_idx:
                q_low = np.percentile(X_arr[:, idx], self.lower_quantile * 100)
                q_high = np.percentile(X_arr[:, idx], self.upper_quantile * 100)
                self.bounds_[idx] = (q_low, q_high)
        return self

    def transform(self, X):
        X_out = X.copy()
        if isinstance(X_out, pd.DataFrame):
            for col, (q_low, q_high) in self.bounds_.items():
                if col in X_out.columns:
                    X_out[col] = np.clip(X_out[col], q_low, q_high)
        else:
            for idx, (q_low, q_high) in self.bounds_.items():
                X_out[:, idx] = np.clip(X_out[:, idx], q_low, q_high)
        return X_out


class TimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Extracts time-of-day cyclical patterns from elapsed seconds without data leakage.
    Dataset Time is elapsed seconds over ~48 hours (0 to 172792s).
    """
    def __init__(self, drop_original=False):
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if 'Time' in X_out.columns:
            # 86400 seconds in a day
            hour_of_day = (X_out['Time'] % 86400) / 3600.0
            X_out['Hour'] = hour_of_day
            X_out['Hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24.0)
            X_out['Hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24.0)
            if self.drop_original:
                X_out = X_out.drop(columns=['Time'])
        return X_out


def create_preprocessor():
    """
    Creates a full scikit-learn preprocessing pipeline.
    - Amount is scaled using RobustScaler (resilient to heavy-tailed distributions).
    - V1 to V28 and Time are capped at 0.1% and 99.9% percentiles to mitigate outlier distortion.
    """
    pca_features = [f"V{i}" for i in range(1, 29)]
    all_features = ['Time'] + pca_features + ['Amount']
    
    # We apply outlier capping to all continuous features
    # RobustScaler to Amount and Time
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
    
    return preprocessor, all_features
