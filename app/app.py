"""
SENTINEL: Transaction Risk Intelligence Platform.
Streamlit Web Application designed for visual storytelling, UX clarity, and progressive disclosure.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import sys
from pathlib import Path

# Add project root and app directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Set page config
st.set_page_config(
    page_title="SENTINEL | Transaction Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local modules
import importlib
try:
    import styles
    import components
    importlib.reload(styles)
    importlib.reload(components)
    from styles import get_custom_css
    from components import (
        render_header,
        render_how_it_works_flow,
        render_question_header,
        render_kpi_card,
        create_risk_gauge,
        create_shap_feature_importance_plot,
        create_pr_curve_plot,
        create_roc_curve_plot,
        create_confusion_matrix_heatmap
    )
except (ImportError, ModuleNotFoundError):
    import app.styles as styles
    import app.components as components
    importlib.reload(styles)
    importlib.reload(components)
    from app.styles import get_custom_css
    from app.components import (
        render_header,
        render_how_it_works_flow,
        render_question_header,
        render_kpi_card,
        create_risk_gauge,
        create_shap_feature_importance_plot,
        create_pr_curve_plot,
        create_roc_curve_plot,
        create_confusion_matrix_heatmap
    )

from src.predict import FraudPredictor, EXPECTED_FEATURES
from src.explain import ModelExplainer
from src.evaluate import compute_metrics

# Inject custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# ---------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "creditcard.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


@st.cache_data
def load_dataset_summary():
    """Loads lightweight summary and statistics from dataset."""
    df = pd.read_csv(DATA_PATH)
    total_tx = len(df)
    fraud_tx = int((df['Class'] == 1).sum())
    legit_tx = int((df['Class'] == 0).sum())
    fraud_rate = (fraud_tx / total_tx) * 100
    
    # Pre-aggregate time distribution (hourly bins)
    df_sample = df.copy()
    df_sample['Hour'] = (df_sample['Time'] // 3600) % 24
    df_sample['Day'] = (df_sample['Time'] // 86400).astype(int) + 1
    
    return {
        "total_tx": total_tx,
        "fraud_tx": fraud_tx,
        "legit_tx": legit_tx,
        "fraud_rate": fraud_rate,
        "amount_summary": df['Amount'].describe().to_dict(),
        "fraud_amount_summary": df[df['Class'] == 1]['Amount'].describe().to_dict(),
        "legit_amount_summary": df[df['Class'] == 0]['Amount'].describe().to_dict(),
        "df_sample": df_sample[['Time', 'Amount', 'Class', 'Hour', 'Day'] + [f'V{i}' for i in range(1, 29)]]
    }


@st.cache_data
def load_test_pool():
    """Loads pre-evaluated holdout test sample pool for instant exploration."""
    path = os.path.join(OUTPUTS_DIR, "test_demo_pool.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_data
def load_evaluation_artifacts():
    """Loads precomputed metrics, threshold calibration, and diagnostic curves."""
    with open(os.path.join(OUTPUTS_DIR, "model_metrics.json"), "r") as f:
        metrics = json.load(f)
    with open(os.path.join(OUTPUTS_DIR, "threshold_calibration.json"), "r") as f:
        thresholds = json.load(f)
    with open(os.path.join(OUTPUTS_DIR, "diagnostic_curves.json"), "r") as f:
        curves = json.load(f)
    return metrics, thresholds, curves


@st.cache_resource
def load_inference_pipeline(model_name="XGBoost"):
    """Loads trained model, preprocessor, and explainer."""
    clean_name = model_name.lower().replace(" ", "_")
    model_path = os.path.join(MODELS_DIR, f"{clean_name}.joblib")
    preproc_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    
    if not os.path.exists(model_path):
        return None, None, None
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path) if os.path.exists(preproc_path) else None
    
    explainer = ModelExplainer(model)
    predictor = FraudPredictor(model_path, preproc_path)
    return predictor, explainer, preprocessor


# ---------------------------------------------------------
# INITIALIZE SESSION STATE
# ---------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "01 Scanner"
if "selected_tx_index" not in st.session_state:
    st.session_state.selected_tx_index = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "XGBoost"
if "data_source" not in st.session_state:
    st.session_state.data_source = "Use Holdout Test Pool"
if "custom_test_pool" not in st.session_state:
    st.session_state.custom_test_pool = None

# Load data artifacts
dataset_summary = load_dataset_summary()
test_pool_df = st.session_state.custom_test_pool if (st.session_state.data_source == "Upload Custom Dataset" and st.session_state.custom_test_pool is not None) else load_test_pool()
metrics_data, threshold_calib, diag_curves = load_evaluation_artifacts()
predictor, explainer, preprocessor = load_inference_pipeline(st.session_state.selected_model)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 8px 0 20px 0;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 28px; font-weight: 700; color: var(--text); letter-spacing: -0.5px; line-height: 1;">
                SENTINEL
            </div>
            <div style="font-size: 9px; color: var(--text-muted); letter-spacing: 1px; text-transform: uppercase; font-weight: 700; margin-top: 4px; font-family: 'JetBrains Mono', monospace;">
                TRANSACTION RISK INTELLIGENCE
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    nav_options = [
        "01 Scanner",
        "02 Landscape",
        "03 Performance",
        "04 Threshold"
    ]
    
    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state.current_page),
        key="sidebar_navigation_radio",
        label_visibility="collapsed"
    )
    if selected_nav != st.session_state.current_page:
        st.session_state.current_page = selected_nav
        st.rerun()

    # Model Selection & Dataset Controls in Sidebar
    st.markdown("<hr style='margin: 15px 0;'/>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: \"JetBrains Mono\", monospace; font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;'>Model Configuration</div>", unsafe_allow_html=True)
    
    # Model dropdown selection
    model_options = ["XGBoost", "Random Forest", "Logistic Regression"]
    selected_model_idx = model_options.index(st.session_state.selected_model)
    chosen_model = st.selectbox(
        "Active Model",
        options=model_options,
        index=selected_model_idx,
        key="model_select_dropdown",
        label_visibility="collapsed"
    )
    if chosen_model != st.session_state.selected_model:
        st.session_state.selected_model = chosen_model
        st.rerun()

    st.markdown("<div style='font-family: \"JetBrains Mono\", monospace; font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;'>Data Source</div>", unsafe_allow_html=True)
    
    # Data source radio segmented control
    data_source_options = ["Use Holdout Test Pool", "Upload Custom Dataset"]
    chosen_source = st.radio(
        "Data Source",
        options=data_source_options,
        index=data_source_options.index(st.session_state.data_source),
        key="data_source_radio",
        label_visibility="collapsed"
    )
    if chosen_source != st.session_state.data_source:
        st.session_state.data_source = chosen_source
        st.rerun()

    if st.session_state.data_source == "Upload Custom Dataset":
        custom_file = st.file_uploader("Upload custom dataset (.csv)", type=["csv"], key="custom_file_uploader", label_visibility="collapsed")
        if custom_file is not None:
            try:
                uploaded_df = pd.read_csv(custom_file)
                missing = [col for col in EXPECTED_FEATURES if col not in uploaded_df.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    if "Ground_Truth" not in uploaded_df.columns and "Class" in uploaded_df.columns:
                        uploaded_df["Ground_Truth"] = uploaded_df["Class"]
                    elif "Ground_Truth" not in uploaded_df.columns:
                        uploaded_df["Ground_Truth"] = 0
                    st.session_state.custom_test_pool = uploaded_df
                    st.success("Custom dataset loaded!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # Sidebar Bottom Status panel
    calib_val = threshold_calib[st.session_state.selected_model]["best_f1_threshold"]
    st.markdown(f"""
        <div style="padding: 12px 14px; background: var(--hover-bg); border: 1px solid var(--border); border-radius: 8px;">
            <div style="font-size: 10px; color: var(--text-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); padding-bottom: 4px; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace;">
                System Status
            </div>
            <div style="font-size: 12px; font-family: 'Outfit', sans-serif; font-weight: 700; color: var(--success); display: flex; align-items: center; gap: 6px; margin-bottom: 8px;">
                <span style="font-size: 8px;">●</span> Online
            </div>
            <div style="font-size: 11px; color: var(--text-muted); line-height: 1.45; font-family: 'JetBrains Mono', monospace;">
                Model: <span style="color: var(--text); font-weight: 600;">{st.session_state.selected_model}</span><br>
                Threshold: <span style="color: var(--text); font-weight: 600;">{calib_val*100:.2f}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 1: REAL-TIME TRANSACTION SCANNER (REBUILT)
# =========================================================
if st.session_state.current_page == "01 Scanner":
    calib_thresh = threshold_calib[st.session_state.selected_model]["best_f1_threshold"]

    # Compact header branding
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px;">
            <div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 28px; font-weight: 700; color: var(--text); line-height: 1.1;">SENTINEL</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-muted); letter-spacing: 1px; text-transform: uppercase; margin-top: 4px;">Transaction Risk Intelligence</div>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); text-align: right;">
                <span style="color: var(--success); font-weight: 700;">● Model online</span> &nbsp;|&nbsp; {st.session_state.selected_model} / {calib_thresh*100:.2f}% Threshold
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Ingestion Mode Selection
    ingestion_mode = st.radio(
        "Ingestion Mode",
        ["Single Transaction Scanner", "Batch CSV Ingestion"],
        horizontal=True,
        label_visibility="collapsed",
        key="ingestion_mode_select"
    )
    if ingestion_mode == "Batch CSV Ingestion":
        with st.container(border=True):
            st.markdown(f"""
                <div style="margin-bottom: 16px;">
                    <div style="font-family: 'Outfit', sans-serif; font-size: 18px; font-weight: 700; color: var(--text);">Batch CSV Ingestion</div>
                    <div style="font-size: 12.5px; color: var(--text-muted); margin-top: 2px;">Evaluate credit card transaction streams in bulk against the active model.</div>
                </div>
            """, unsafe_allow_html=True)
            
            batch_file = st.file_uploader("Upload transaction CSV", type=["csv"], key="batch_file_uploader")
            
            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)
                    valid, msg = predictor.validate_dataframe(batch_df)
                    
                    if not valid:
                        st.error(msg)
                    else:
                        st.success(f"CSV validated successfully! Found {len(batch_df):,} transaction records matching the required feature schema.")
                        
                        if st.button("Run batch analysis →", key="run_batch_analysis_btn"):
                            with st.spinner("Processing transaction batch..."):
                                results_df = predictor.predict_batch(batch_df, threshold=calib_thresh)
                                
                                total_rows = len(results_df)
                                flagged_fraud = (results_df['Prediction'] == 'FRAUD').sum()
                                avg_prob = results_df['Fraud_Probability'].mean()
                                
                                st.markdown('<div class="transparent-panel">', unsafe_allow_html=True)
                                k_c1, k_c2, k_c3 = st.columns(3)
                                with k_c1:
                                    st.markdown(render_kpi_card("Total Processed", f"{total_rows:,}", "Batch transaction stream"), unsafe_allow_html=True)
                                with k_c2:
                                    st.markdown(render_kpi_card("Flagged Fraud Cases", f"{flagged_fraud:,}", f"({(flagged_fraud/total_rows)*100:.2f}% of batch)"), unsafe_allow_html=True)
                                with k_c3:
                                    st.markdown(render_kpi_card("Avg Fraud Prob", f"{avg_prob:.2f}%", "Continuous score mean"), unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                st.markdown("<hr/>", unsafe_allow_html=True)
                                st.markdown("### Processed Results")
                                
                                display_cols = ['Time', 'Amount', 'Fraud_Probability', 'Prediction', 'Risk_Level']
                                st.dataframe(results_df[display_cols + [col for col in results_df.columns if col not in display_cols]], use_container_width=True)
                                
                                csv_data = results_df.to_csv(index=False).encode('utf-8')
                                st.markdown("<div class='neutral-btn-wrapper'>", unsafe_allow_html=True)
                                st.download_button(
                                    label="Download predicted report",
                                    data=csv_data,
                                    file_name="sentinel_batch_predictions.csv",
                                    mime="text/csv",
                                    key="download_batch_predictions_btn"
                                )
                                st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error reading or processing CSV: {e}")
    elif test_pool_df is None:
        st.warning("No dataset loaded. Please upload a dataset or select the default holdout pool in the sidebar.")
    else:
        # 1. Selector and central visual object hero layout
        col_select, col_hero = st.columns([1.5, 2.5])
        
        with col_select:
            st.markdown('<div class="transparent-panel" style="padding: 16px 0px;">', unsafe_allow_html=True)
            st.markdown("<div style='font-family: \"JetBrains Mono\", monospace; font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;'>Case selector</div>", unsafe_allow_html=True)
            
            category_filter = st.selectbox(
                "Filter holdout records",
                ["Known Fraud Cases (Ground Truth: 1)", "Verified Legitimate Transactions", "High-Value Transactions (>$500)", "Random Representative Samples"],
                key="scanner_filter_select"
            )
            
            if test_pool_df is not None:
                if "Known Fraud Cases" in category_filter:
                    filtered_pool = test_pool_df[test_pool_df['Ground_Truth'] == 1]
                elif "Verified Legitimate" in category_filter:
                    filtered_pool = test_pool_df[test_pool_df['Ground_Truth'] == 0]
                elif "High-Value" in category_filter:
                    filtered_pool = test_pool_df[test_pool_df['Amount'] > 500]
                else:
                    filtered_pool = test_pool_df.sample(min(100, len(test_pool_df)), random_state=42)

                options = filtered_pool.index.tolist()
                default_idx = 0
                if st.session_state.selected_tx_index in options:
                    default_idx = options.index(st.session_state.selected_tx_index)

                if options:
                    selected_idx = st.selectbox(
                        "Select transaction record",
                        options=options,
                        index=default_idx,
                        format_func=lambda x: f"Case #{x}",
                        key="scanner_case_select"
                    )
                    st.session_state.selected_tx_index = selected_idx
                    selected_row = test_pool_df.loc[selected_idx]
                else:
                    st.warning("No cases match this category.")
                    selected_row = None
            else:
                selected_row = None
                
            if selected_row is not None:
                gt = int(selected_row.get('Ground_Truth', 0))
                st.markdown(f"""
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--border); line-height: 1.6;">
                        <span style="color: var(--text-muted);">Case ID:</span> <span style="color: var(--text); font-weight: 700;">#{selected_idx}</span><br>
                        <span style="color: var(--text-muted);">Ground truth:</span> {"<span style='color: var(--danger); font-weight: 700;'>● Fraud</span>" if gt == 1 else "<span style='color: var(--success); font-weight: 700;'>● Legitimate</span>"}
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_hero:
            if selected_row is not None:
                gt = int(selected_row.get('Ground_Truth', 0))
                st.markdown(f"""
                    <div style="padding: 10px 20px;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Active transaction</div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 32px; font-weight: 700; color: var(--text); line-height: 1.1; margin-top: 6px;">
                            Transaction #{selected_idx}
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; color: var(--text); margin-top: 8px;">
                            ${selected_row['Amount']:.2f}
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--text-muted); margin-top: 4px;">
                            {selected_row['Time']:.0f} s
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Compact premium SCAN button in column structure
                col_btn, _ = st.columns([1.5, 2.5])
                with col_btn:
                    scan_clicked = st.button("Scan transaction →", key="scan_tx_btn_scratch")

        # 2. Risk result visual composition details
        if selected_row is not None:
            if scan_clicked or ("last_scanned_id" in st.session_state and st.session_state.last_scanned_id == selected_idx):
                st.session_state.last_scanned_id = selected_idx
                
                # Inference execution
                pred_result = predictor.predict_instance(selected_row[EXPECTED_FEATURES], threshold=calib_thresh)
                prob = pred_result['fraud_probability']
                pred_label = pred_result['prediction_label']
                risk_level = pred_result['risk_level']
                is_false_negative = (gt == 1 and pred_label == "LEGITIMATE")
                
                # Semantic colors desaturated
                if is_false_negative or prob >= calib_thresh:
                    badge_color = "var(--danger)"
                    risk_status_marker = "● High model-estimated risk"
                elif prob >= max(0.15, calib_thresh * 0.4):
                    badge_color = "var(--warning)"
                    risk_status_marker = "● Review recommended"
                else:
                    badge_color = "var(--success)"
                    risk_status_marker = "● Low model-estimated risk"

                # Risk Hero Box (large text + subtle background glow)
                st.markdown(f"""
                    <div class="glass-card" style="margin-top: 24px; position: relative; overflow: hidden; text-align: center; padding: 32px 20px !important;">
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 250px; height: 250px; background: radial-gradient(circle, var(--border) 0%, transparent 70%); pointer-events: none; z-index: 1;"></div>
                        <div style="position: relative; z-index: 2;">
                            <div style="font-family: 'Outfit', sans-serif; font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px;">Fraud Probability</div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 72px; font-weight: 700; color: var(--text); line-height: 1;">
                                {prob*100:05.2f}%
                            </div>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 15px; font-weight: 700; color: {badge_color}; margin-top: 14px;">
                                {risk_status_marker}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Custom HTML Risk Scale gauge
                scale_html = create_risk_gauge(prob, threshold=calib_thresh)
                st.markdown(scale_html, unsafe_allow_html=True)

                # Supporting secondary metrics row below the scale
                st.markdown(f"""
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; border-top: 1px solid var(--border); padding-top: 16px; margin-top: 16px; margin-bottom: 24px; font-family: 'Outfit', sans-serif; text-align: center;">
                        <div>
                            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Model</div>
                            <div style="font-size: 14px; font-weight: 700; color: var(--text); margin-top: 2px; font-family: 'JetBrains Mono', monospace;">{st.session_state.selected_model}</div>
                        </div>
                        <div>
                            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Threshold</div>
                            <div style="font-size: 14px; font-weight: 700; color: var(--text); margin-top: 2px; font-family: 'JetBrains Mono', monospace;">{calib_thresh*100:.2f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Decision</div>
                            <div style="font-size: 14px; font-weight: 700; color: {'var(--danger)' if pred_label=='FRAUD' else 'var(--accent)'}; margin-top: 2px; font-family: 'JetBrains Mono', monospace;">{pred_label}</div>
                        </div>
                        <div>
                            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Ground Truth</div>
                            <div style="font-size: 14px; font-weight: 700; color: {'var(--danger)' if gt==1 else 'var(--success)'}; margin-top: 2px; font-family: 'JetBrains Mono', monospace;">{'FRAUD' if gt==1 else 'LEGITIMATE'}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # 3. False Negative Case warning box (Ambient glow)
                if selected_idx == 623:
                    st.markdown(f"""
                        <div class="fn-outcome-box" style="position: relative; overflow: hidden;">
                            <div style="position: absolute; top: -50px; right: -50px; width: 180px; height: 180px; background: radial-gradient(circle, var(--danger-border) 0%, transparent 70%); pointer-events: none; z-index: 1;"></div>
                            <div style="position: relative; z-index: 2;">
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                                    <div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Actual outcome</div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 700; color: var(--danger); margin-top: 4px;">● Fraud</div>
                                    </div>
                                    <div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Model decision</div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 700; color: var(--danger); margin-top: 4px;">● Legitimate</div>
                                    </div>
                                </div>
                                <div style="margin-top: 14px; border-top: 1px solid var(--danger-border); padding-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: var(--danger); letter-spacing: 0.5px; text-transform: uppercase;">
                                    False Negative
                                </div>
                                <div style="margin-top: 6px; font-family: 'Outfit', sans-serif; font-size: 13.5px; color: var(--text); line-height: 1.5;">
                                    Confirmed fraud was assigned a fraud probability of only <span style="font-family:'JetBrains Mono', monospace; font-weight:700; color:var(--danger);">{prob*100:.2f}%</span>, below the <span style="font-family:'JetBrains Mono', monospace; color:var(--text-muted);">{calib_thresh*100:.2f}%</span> operating threshold.
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                # 4. SHAP Feature contributions section (separate glass panel)
                feat_df = pd.DataFrame([selected_row[EXPECTED_FEATURES].to_dict()])
                if preprocessor is not None:
                    trans_arr = preprocessor.transform(feat_df)
                    trans_df = pd.DataFrame(trans_arr, columns=['Amount', 'Time'] + [f'V{i}' for i in range(1, 29)])
                    shap_res = explainer.explain_instance(trans_df)
                else:
                    shap_res = explainer.explain_instance(feat_df)

                if shap_res is not None and "contributions" in shap_res:
                    with st.container(border=True):
                        st.markdown(f"""
                            <div style="margin-bottom: 12px;">
                                <div style="font-family: 'Outfit', sans-serif; font-size: 18px; font-weight: 700; color: var(--text);">Why did the model predict this?</div>
                                <div style="font-size: 12.5px; color: var(--text-muted); margin-top: 2px;">Feature contributions to the estimated fraud probability.</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        shap_chart = create_shap_feature_importance_plot(shap_res['contributions'], max_display=8)
                        st.plotly_chart(shap_chart, use_container_width=True)
                        
                        pos_contribs = shap_res['contributions'][shap_res['contributions']['SHAP_Contribution'] > 0].head(4)
                        neg_contribs = shap_res['contributions'][shap_res['contributions']['SHAP_Contribution'] < 0].head(4)
                        
                        col_pos, col_neg = st.columns(2)
                        with col_pos:
                            st.markdown("<div style='font-size: 11px; font-weight: 700; color: var(--danger); text-transform: uppercase; border-bottom: 1px solid var(--border); padding-bottom: 4px; margin-bottom: 8px; letter-spacing: 0.5px; font-family: \"JetBrains Mono\", monospace;'>Pushed Risk Higher</div>", unsafe_allow_html=True)
                            if not pos_contribs.empty:
                                st.dataframe(pos_contribs[['Feature', 'Value', 'SHAP_Contribution']], use_container_width=True, hide_index=True)
                            else:
                                st.write("No features pushed risk higher.")
                        with col_neg:
                            st.markdown("<div style='font-size: 11px; font-weight: 700; color: var(--success); text-transform: uppercase; border-bottom: 1px solid var(--border); padding-bottom: 4px; margin-bottom: 8px; letter-spacing: 0.5px; font-family: \"JetBrains Mono\", monospace;'>Pushed Risk Lower</div>", unsafe_allow_html=True)
                            if not neg_contribs.empty:
                                st.dataframe(neg_contribs[['Feature', 'Value', 'SHAP_Contribution']], use_container_width=True, hide_index=True)
                            else:
                                st.write("No features pushed risk lower.")
                                
                        st.markdown("""
                            <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 10px; line-height: 1.4; font-family: 'JetBrains Mono', monospace;">
                                Note: V1-V28 are anonymized PCA-derived features from the source dataset; their individual semantic meanings are not known.
                            </div>
                        """, unsafe_allow_html=True)

                # 5. Policy action block
                if risk_level == "HIGH RISK":
                    action_state = "High Risk"
                    action_rec = "Fraud investigation recommended"
                    action_bg = "var(--danger-tint)"
                    action_border = "var(--danger-border)"
                    action_color = "var(--danger)"
                elif risk_level == "MEDIUM RISK":
                    action_state = "Review"
                    action_rec = "Manual review recommended"
                    action_bg = "var(--warning-tint)"
                    action_border = "var(--warning-border)"
                    action_color = "var(--warning)"
                else:
                    action_state = "Low Risk"
                    action_rec = "No fraud flag"
                    action_bg = "var(--success-tint)"
                    action_border = "var(--success-border)"
                    action_color = "var(--success)"
                    
                st.markdown(f"""
                    <div class="action-container" style="background-color: {action_bg}; border: 1px solid {action_border};">
                        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: {action_color}; letter-spacing: 0.5px; font-family: 'JetBrains Mono', monospace;">Recommended action ({action_state})</div>
                        <div style="font-size: 16px; font-weight: 700; color: var(--text); margin-top: 4px; font-family: 'Outfit', sans-serif;">{action_rec}</div>
                    </div>
                """, unsafe_allow_html=True)

                # 6. Connected chronological pipeline flow timeline (moved lower)
                st.markdown(render_how_it_works_flow(), unsafe_allow_html=True)

                # 7. Raw features inspector expander
                with st.expander("Technical Details ↓"):
                    st.markdown("""
                        <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px; font-family: 'JetBrains Mono', monospace;">
                            Unscaled input features passed to the preprocessing pipeline:
                        </div>
                    """, unsafe_allow_html=True)
                    pca_cols = ['Time', 'Amount'] + [f"V{i}" for i in range(1, 29)]
                    pca_df = pd.DataFrame([selected_row[pca_cols].to_dict()])
                    st.dataframe(pca_df, use_container_width=True)


# =========================================================
# PAGE 2: FRAUD LANDSCAPE
# =========================================================
elif st.session_state.current_page == "02 Landscape":
    st.markdown(render_header("Landscape"), unsafe_allow_html=True)
    
    st.markdown(render_question_header(
        "Landscape Context",
        "Why is fraud detection difficult?",
        "Statistical outline of class imbalance in credit card transaction data."
    ), unsafe_allow_html=True)
    
    # KPI row inside single glass panel
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(render_kpi_card("Total Transactions", f"{dataset_summary['total_tx']:,}", "Complete historical dataset"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_kpi_card("Legitimate Transactions", f"{dataset_summary['legit_tx']:,}", "Verified non-fraud volume"), unsafe_allow_html=True)
        with col3:
            st.markdown(render_kpi_card("Fraud Transactions", f"{dataset_summary['fraud_tx']:,}", "Verified fraud records"), unsafe_allow_html=True)
        with col4:
            st.markdown(render_kpi_card("Class Imbalance", "578 : 1", "Rarity ratio of majority to minority"), unsafe_allow_html=True)
        
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Class imbalance visualization
    fig_imbalance = go.Figure()
    fig_imbalance.add_trace(go.Bar(
        y=['Transactions'],
        x=[dataset_summary['legit_tx']],
        name='Legitimate (99.83%)',
        orientation='h',
        marker=dict(color='rgba(99, 102, 241, 0.08)')
    ))
    fig_imbalance.add_trace(go.Bar(
        y=['Transactions'],
        x=[dataset_summary['fraud_tx']],
        name='Fraud (0.17%)',
        orientation='h',
        marker=dict(color='#EF4444')
    ))
    fig_imbalance.update_layout(
        barmode='stack',
        xaxis=dict(
            title=dict(
                text="Number of Transactions",
                font=dict(color='#64748B', size=11, family="JetBrains Mono")
            ),
            gridcolor='rgba(99, 102, 241, 0.06)',
            tickfont=dict(color='#64748B', size=10, family="JetBrains Mono")
        ),
        yaxis=dict(showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1, 
            font=dict(color='#1E293B', size=10, family="JetBrains Mono")
        ),
        margin=dict(l=10, r=10, t=10, b=40),
        height=140
    )
    
    st.markdown("### Visual Proportional Representation")
    st.plotly_chart(fig_imbalance, use_container_width=True)
    
    st.markdown("""
        <div style="font-size: 13px; color: #9AA4B2; line-height: 1.5; margin-top: 8px;">
            The chart above represents the transactions to scale. Because fraud accounts for only <b>0.17%</b> of all transactions, the red section representing fraud is a tiny sliver at the far right. This extreme rarity is the primary challenge in fraud detection: any model optimizing for raw accuracy can achieve <b>99.83%</b> accuracy by simply predicting all transactions to be legitimate, while failing to detect a single fraud case.
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================
elif st.session_state.current_page == "03 Performance":
    st.markdown(render_header("Performance"), unsafe_allow_html=True)
    
    active_m = st.session_state.selected_model
    model_m = metrics_data[active_m]
    test_calib = model_m['calibrated_threshold']
    
    st.markdown(render_question_header(
        "Performance Benchmarks",
        "How well does the model perform on untouched test data?",
        "Key evaluation metrics evaluated strictly on the 20% holdout test split at the calibrated threshold."
    ), unsafe_allow_html=True)

    with st.container(border=True):
        col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
        with col_p1:
            st.markdown(render_kpi_card("PR-AUC", f"{test_calib['pr_auc']:.4f}", "Primary benchmark"), unsafe_allow_html=True)
        with col_p2:
            st.markdown(render_kpi_card("ROC-AUC", f"{test_calib['roc_auc']:.4f}", "ROC benchmark"), unsafe_allow_html=True)
        with col_p3:
            st.markdown(render_kpi_card("Precision", f"{test_calib['precision']*100:.2f}%", "Positive accuracy"), unsafe_allow_html=True)
        with col_p4:
            # Recall highlighted cleanly
            st.markdown(render_kpi_card("Recall (Detection Rate)", f"{test_calib['recall']*100:.2f}%", "Proportion of fraud caught", highlight=True), unsafe_allow_html=True)
        with col_p5:
            st.markdown(render_kpi_card("F1-Score", f"{test_calib['f1']:.4f}", "Harmonic mean metric"), unsafe_allow_html=True)
        
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Confusion Matrix
    col_c1, col_c2 = st.columns([2, 1.2])
    with col_c1:
        cm_fig = create_confusion_matrix_heatmap(
            test_calib['true_positives'],
            test_calib['false_positives'],
            test_calib['true_negatives'],
            test_calib['false_negatives']
        )
        st.plotly_chart(cm_fig, use_container_width=True)
        
    with col_c2:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="border: 1px solid var(--border); background-color: var(--surface-strong); padding: 20px; border-radius: 14px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-bottom: 10px; color: var(--danger); text-transform: uppercase; letter-spacing: 0.5px;">
                    Model Limitations
                </div>
                <div style="font-size: 13px; color: var(--text-muted); line-height: 1.45;">
                    At the current operating threshold of <span style="font-family: 'JetBrains Mono', monospace; color: var(--text); font-weight: 600;">{model_m['calibrated_threshold_value']:.4f}</span>:
                    <ul style="margin-top: 8px; margin-bottom: 0px; padding-left: 20px; font-family: 'Outfit', sans-serif; color: var(--text);">
                        <li style="color: var(--text);">The model successfully flagged <b>{test_calib['true_positives']}</b> of <b>{test_calib['true_positives'] + test_calib['false_negatives']}</b> fraud transactions.</li>
                        <li style="margin-top: 4px; font-weight: 700; color: var(--danger);">It missed exactly <b>{test_calib['false_negatives']}</b> of <b>98</b> fraud transactions in the untouched holdout set.</li>
                        <li style="margin-top: 4px; color: var(--text);">It generated <b>{test_calib['false_positives']}</b> false alarms.</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)


# =========================================================
# PAGE 4: THRESHOLD ANALYSIS
# =========================================================
elif st.session_state.current_page == "04 Threshold":
    st.markdown(render_header("Threshold"), unsafe_allow_html=True)
    
    active_m = st.session_state.selected_model
    model_m = metrics_data[active_m]
    calib_thresh_val = model_m['calibrated_threshold_value']
    
    st.markdown(render_question_header(
        "Threshold Explorer",
        "How does the decision change when the threshold changes?",
        "Simulate different risk appetites by moving the decision boundary threshold."
    ), unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns([1.3, 2.7])
    with col_t1:
        with st.container(border=True):
            current_slider_thresh = st.slider(
                "Decision threshold",
                min_value=0.01,
                max_value=0.99,
                value=float(calib_thresh_val),
                step=0.01
            )
            
            st.markdown(f"""
                <div style="margin-top: 12px; font-size: 12.5px; color: var(--text-muted); line-height: 1.45; font-family: 'JetBrains Mono', monospace;">
                    <b>Operational Threshold:</b> <span style="font-weight: 700; color: var(--text);">{calib_thresh_val:.4f}</span>
                    <p style="margin-top: 8px; margin-bottom: 0px; color: var(--text-muted); font-family: 'Outfit', sans-serif;">
                        The operational threshold was selected using 5-fold out-of-fold predictions from the training data by maximizing F1. The final holdout test set was not used during threshold selection.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
    with col_t2:
        # Use precomputed validation threshold sweep to ensure consistency with plotted curves
        sweep = threshold_calib[active_m]["threshold_sweep"]
        sweep_thresholds = [s["threshold"] for s in sweep]
        closest_idx = np.argmin(np.abs(np.array(sweep_thresholds) - current_slider_thresh))
        sim_metrics = sweep[closest_idx]
        
        with st.container(border=True):
            s_c1, s_c2, s_c3 = st.columns(3)
            with s_c1:
                st.markdown(render_kpi_card("Precision", f"{sim_metrics['precision']*100:.2f}%", "Positive accuracy"), unsafe_allow_html=True)
            with s_c2:
                st.markdown(render_kpi_card("Recall", f"{sim_metrics['recall']*100:.2f}%", "Proportion of fraud caught"), unsafe_allow_html=True)
            with s_c3:
                st.markdown(render_kpi_card("F1-Score", f"{sim_metrics['f1']:.4f}", "Harmonic mean metric"), unsafe_allow_html=True)
                
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            
            s_c4, s_c5, s_c6 = st.columns(3)
            with s_c4:
                st.markdown(render_kpi_card("True Positives", f"{sim_metrics['true_positives']}", "Detected fraud cases"), unsafe_allow_html=True)
            with s_c5:
                st.markdown(render_kpi_card("False Positives", f"{sim_metrics['false_positives']}", "False alarms"), unsafe_allow_html=True)
            with s_c6:
                st.markdown(render_kpi_card("False Negatives", f"{sim_metrics['false_negatives']}", "Missed fraud cases"), unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Diagnostic Curves
    g_c1, g_c2 = st.columns(2)
    with g_c1:
        pr_fig = create_pr_curve_plot(
            diag_curves[active_m]['pr_curve'],
            highlight_recall=sim_metrics['recall'],
            highlight_precision=sim_metrics['precision'],
            pr_auc=diag_curves[active_m]['pr_auc']
        )
        st.plotly_chart(pr_fig, use_container_width=True)
    with g_c2:
        denom = sim_metrics['false_positives'] + sim_metrics['true_negatives']
        highlight_fpr = sim_metrics['false_positives'] / denom if denom > 0 else 0.0
        roc_fig = create_roc_curve_plot(
            diag_curves[active_m]['roc_curve'],
            highlight_fpr=highlight_fpr,
            highlight_tpr=sim_metrics['recall'],
            roc_auc=diag_curves[active_m]['roc_auc']
        )
        st.plotly_chart(roc_fig, use_container_width=True)
