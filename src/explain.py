"""
Explainability Module using SHAP.
Extracts mathematically rigorous feature contributions for transactions.
Strictly adheres to dataset feature names (Time, V1..V28, Amount) without fabricated business labels.
"""

import numpy as np
import pandas as pd
import shap


class ModelExplainer:
    """
    Computes local and global feature attribution values using SHAP.
    """
    def __init__(self, model, background_data=None):
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        try:
            # TreeExplainer for Tree-based models (XGBoost / RandomForest)
            if hasattr(self.model, "estimators_") or "XGB" in str(type(self.model)):
                self.explainer = shap.TreeExplainer(self.model)
            elif self.background_data is not None:
                # Linear / Kernel Explainer with representative background
                bg_sample = shap.sample(self.background_data, min(100, len(self.background_data)))
                self.explainer = shap.Explainer(self.model, bg_sample)
            else:
                self.explainer = shap.Explainer(self.model)
        except Exception:
            # Fallback to general explainer
            if self.background_data is not None:
                bg_sample = shap.sample(self.background_data, min(50, len(self.background_data)))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, bg_sample)
            else:
                self.explainer = None

    def explain_instance(self, feature_vector_df):
        """
        Computes SHAP values for a single transaction.
        Returns a sorted DataFrame with feature names, raw values, and contribution impacts.
        """
        if self.explainer is None:
            return None
            
        try:
            shap_values = self.explainer(feature_vector_df)
            
            # Handle multi-output (classification [class 0, class 1])
            if len(shap_values.shape) == 3: # (instances, features, classes)
                values = shap_values.values[0, :, 1]
                base_val = shap_values.base_values[0, 1] if hasattr(shap_values.base_values, "__len__") else shap_values.base_values
            else:
                values = shap_values.values[0]
                base_val = shap_values.base_values[0] if hasattr(shap_values.base_values, "__len__") else shap_values.base_values

            feature_names = feature_vector_df.columns.tolist()
            raw_vals = feature_vector_df.iloc[0].tolist()

            df_contrib = pd.DataFrame({
                "Feature": feature_names,
                "Value": raw_vals,
                "SHAP_Contribution": values,
                "Abs_Impact": np.abs(values)
            }).sort_values(by="Abs_Impact", ascending=False)
            
            return {
                "contributions": df_contrib,
                "base_value": float(base_val)
            }
        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return None
