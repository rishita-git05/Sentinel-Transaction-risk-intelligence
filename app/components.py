"""
Reusable UI and Plotly chart components for SENTINEL platform.
Implements a refined light-first glassmorphism fintech aesthetic.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def render_header(current_page="Transaction Scanner"):
    """
    Renders top product branding bar.
    """
    return f"""<div class="sentinel-header">
    <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 12px;">
        <div>
            <div class="sentinel-title">SENTINEL</div>
            <div class="sentinel-tagline">TRANSACTION RISK INTELLIGENCE</div>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); text-transform: none;">
            Pipeline status: <span style="font-weight: 700; color: var(--success);">Active</span> &nbsp;|&nbsp; View: <span style="font-weight: 700; color: var(--text);">{current_page}</span>
        </div>
    </div>
</div>"""


def render_how_it_works_flow():
    """
    Renders the horizontal 5-step timeline as a connected visual timeline with circles and active status indicators.
    """
    return """<div class="timeline-pipeline">
    <div class="timeline-line"></div>
    <div class="timeline-step active">
        <div class="timeline-circle">01</div>
        <div class="timeline-title">Input</div>
        <div class="timeline-desc">Raw parameters ingested.</div>
    </div>
    <div class="timeline-step active">
        <div class="timeline-circle">02</div>
        <div class="timeline-title">Preprocess</div>
        <div class="timeline-desc">Pipeline scaling applied.</div>
    </div>
    <div class="timeline-step active">
        <div class="timeline-circle">03</div>
        <div class="timeline-title">Risk score</div>
        <div class="timeline-desc">XGBoost estimates probability.</div>
    </div>
    <div class="timeline-step active">
        <div class="timeline-circle">04</div>
        <div class="timeline-title">Explanation</div>
        <div class="timeline-desc">SHAP attributes signal.</div>
    </div>
    <div class="timeline-step active">
        <div class="timeline-circle">05</div>
        <div class="timeline-title">Decision</div>
        <div class="timeline-desc">Operational threshold matched.</div>
    </div>
</div>"""


def render_question_header(badge_text, question_text, subtitle_text=""):
    """
    Renders a clean technical section header.
    """
    sub_html = f"<div class='section-subtitle' style='margin-top: 4px; margin-bottom: 0px; font-size: 12px; color: var(--text-muted);'>{subtitle_text}</div>" if subtitle_text else ""
    return f"""<div class="question-header">
    <div class="badge-label badge-neutral" style="margin-bottom: 6px; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--accent); border-color: var(--border);">{badge_text}</div>
    <div class="question-title">{question_text}</div>
    {sub_html}
</div>"""


def render_kpi_card(title, value, subtitle, icon="📊", border_color="rgba(51, 65, 85, 0.6)", highlight=False):
    """
    Generates HTML for a sleek, technical KPI block within glass cards.
    """
    title_lower = title.lower()
    if highlight:
        card_class = "kpi-card kpi-highlight"
    elif "fraud" in title_lower or "negative" in title_lower or "flagged" in title_lower:
        card_class = "kpi-card kpi-danger"
    elif "legitimate" in title_lower or "positive" in title_lower or "success" in title_lower or "pr-auc" in title_lower:
        card_class = "kpi-card kpi-success"
    elif "warning" in title_lower or "imbalance" in title_lower or "false" in title_lower or "avg" in title_lower or "average" in title_lower:
        card_class = "kpi-card kpi-warning"
    else:
        card_class = "kpi-card kpi-info"
        
    return f"""<div class="{card_class}">
    <div class="kpi-label">{title}</div>
    <div class="kpi-value">{value}</div>
    <div class="kpi-sub">{subtitle}</div>
</div>"""


def create_risk_gauge(probability, threshold=0.5):
    """
    Returns an HTML/CSS risk spectrum gauge for transaction score details.
    """
    prob_pct = probability * 100
    thresh_pct = threshold * 100
    
    if probability >= threshold:
        color = "var(--danger)"
        color_raw = "#EF4444"
    elif probability >= max(0.15, threshold * 0.4):
        color = "var(--warning)"
        color_raw = "#F59E0B"
    else:
        color = "var(--success)"
        color_raw = "#10B981"
        
    return f"""<div style="position: relative; width: 100%; height: 55px; margin: 20px 0; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
<div style="display: flex; justify-content: space-between; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; margin-bottom: 6px;">
<span>Low risk</span>
<span>High risk</span>
</div>
<div style="position: absolute; top: 18px; left: 0; right: 0; height: 2px; background: var(--border);"></div>
<div style="position: absolute; top: 14px; left: {thresh_pct}%; width: 2px; height: 10px; background: var(--text-muted);" title="Operational Threshold"></div>
<div style="position: absolute; top: 26px; left: {thresh_pct}%; transform: translateX(-50%); font-size: 9px; color: var(--text-muted);">Threshold: {thresh_pct:.2f}%</div>
<div style="position: absolute; top: 14px; left: {prob_pct}%; transform: translateX(-50%); width: 12px; height: 12px; border-radius: 50%; background: {color}; box-shadow: 0 0 10px {color_raw}; transition: all 0.3s ease;"></div>
<div style="position: absolute; top: 28px; left: {prob_pct}%; transform: translateX(-50%); color: {color}; font-weight: 700; font-size: 12px;">{prob_pct:.2f}%</div>
</div>"""


def create_shap_feature_importance_plot(shap_df, max_display=8):
    """
    Creates a clean, minimal horizontal bar chart for SHAP variables.
    """
    top_df = shap_df.head(max_display).copy()
    top_df = top_df.sort_values(by="SHAP_Contribution", ascending=True)
    
    # Emerald green/teal negative contributions, Crimson/coral positive contributions, Neutral slate
    colors = []
    for x in top_df['SHAP_Contribution']:
        if abs(x) < 0.005:
            colors.append('#94A3B8')  # Neutral slate
        elif x > 0:
            colors.append('#EF4444')  # Coral/red (fraud push)
        else:
            colors.append('#0D9488')  # Emerald/teal (legitimate push)
            
    fig = go.Figure(go.Bar(
        x=top_df['SHAP_Contribution'],
        y=top_df['Feature'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(0,0,0,0)', width=0)),
        text=[f" {val:.3f}" for val in top_df['Value']],
        hovertemplate="<b>Feature:</b> %{y}<br><b>Contribution:</b> %{x:.4f}<br><b>Value:</b> %{text}<extra></extra>"
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(
                text="Contribution to Risk Score",
                font=dict(color='#64748B', size=11, family="JetBrains Mono")
            ),
            gridcolor='rgba(99, 102, 241, 0.06)',
            zerolinecolor='rgba(99, 102, 241, 0.15)',
            tickfont=dict(color='#64748B', size=10, family="JetBrains Mono")
        ),
        yaxis=dict(
            tickfont=dict(color='#475569', family='Outfit', size=11),
            gridcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=60, r=20, t=10, b=40),
        height=260
    )
    return fig


def create_pr_curve_plot(pr_data, highlight_recall=None, highlight_precision=None, pr_auc=0.86):
    """
    Plots the Precision-Recall curve with threshold point highlighted.
    """
    prec = pr_data['precision']
    rec = pr_data['recall']

    fig = go.Figure()
    
    # Curve line (Deep Indigo Accent)
    fig.add_trace(go.Scatter(
        x=rec,
        y=prec,
        mode='lines',
        name=f'PR (AUC = {pr_auc:.4f})',
        line=dict(color='#4F46E5', width=2),
        hovertemplate="<b>Recall:</b> %{x:.4f}<br><b>Precision:</b> %{y:.4f}<extra></extra>"
    ))

    # Highlight point (Coral Danger)
    if highlight_recall is not None and highlight_precision is not None:
        fig.add_trace(go.Scatter(
            x=[highlight_recall],
            y=[highlight_precision],
            mode='markers',
            name=f'Selected Threshold',
            marker=dict(color='#EF4444', size=10, symbol='square', line=dict(color='#1E293B', width=1.5)),
            hovertemplate=f"<b>Selected Point</b><br>Recall: {highlight_recall:.4f}<br>Precision: {highlight_precision:.4f}<extra></extra>"
        ))

    # Baseline (Neutral Slate)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0.00173, 0.00173],
        mode='lines',
        name='Baseline (0.17%)',
        line=dict(color='#94A3B8', width=1, dash='dash')
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Precision-Recall Curve</b> (AUC = {pr_auc:.4f})",
            font=dict(size=13, color='#1E293B', family='Outfit')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(
                text="Recall (Sensitivity)",
                font=dict(color='#64748B', size=11, family="JetBrains Mono")
            ),
            range=[0, 1.02],
            gridcolor='rgba(99, 102, 241, 0.06)',
            tickfont=dict(color='#64748B', size=10, family="JetBrains Mono")
        ),
        yaxis=dict(
            title=dict(
                text="Precision (Positive Accuracy)",
                font=dict(color='#64748B', size=11, family="JetBrains Mono")
            ),
            range=[0, 1.05],
            gridcolor='rgba(99, 102, 241, 0.06)',
            tickfont=dict(color='#64748B', size=10, family="JetBrains Mono")
        ),
        legend=dict(
            x=0.02, y=0.15,
            bgcolor='rgba(255, 255, 255, 0.85)',
            bordercolor='rgba(99, 102, 241, 0.12)',
            font=dict(color='#1E293B', size=9, family="JetBrains Mono")
        ),
        margin=dict(l=50, r=20, t=40, b=40),
        height=280
    )
    return fig


def create_roc_curve_plot(roc_data, highlight_fpr=None, highlight_tpr=None, roc_auc=0.98):
    """
    Plots the ROC curve with operating threshold highlighted.
    """
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']

    fig = go.Figure()
    
    # Curve line (Deep Indigo Accent)
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.4f})',
        line=dict(color='#4F46E5', width=2),
        hovertemplate="<b>FPR:</b> %{x:.4f}<br><b>TPR:</b> %{y:.4f}<extra></extra>"
    ))

    # Highlight point
    if highlight_fpr is not None and highlight_tpr is not None:
        fig.add_trace(go.Scatter(
            x=[highlight_fpr],
            y=[highlight_tpr],
            mode='markers',
            name=f'Selected Threshold',
            marker=dict(color='#EF4444', size=10, symbol='square', line=dict(color='#1E293B', width=1.5)),
            hovertemplate=f"<b>Selected Point</b><br>FPR: {highlight_fpr:.4f}<br>TPR: {highlight_tpr:.4f}<extra></extra>"
        ))

    # Diagonal baseline (Neutral Slate)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='#94A3B8', width=1, dash='dash')
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>ROC Curve</b> (AUC = {roc_auc:.4f})",
            font=dict(size=13, color='#1E293B', family='Outfit')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(
                text="False Positive Rate",
                font=dict(color='#64748B', size=11, family="JetBrains Mono")
            ),
            range=[0, 1.02],
            gridcolor='rgba(99, 102, 241, 0.06)',
            tickfont=dict(color='#64748B', size=10, family="JetBrains Mono")
        ),
        yaxis=dict(
            title=dict(
                text="True Positive Rate (Recall)",
                font=dict(color='#64748B', size=11, family="JetBrains Mono")
            ),
            range=[0, 1.05],
            gridcolor='rgba(99, 102, 241, 0.06)',
            tickfont=dict(color='#64748B', size=10, family="JetBrains Mono")
        ),
        legend=dict(
            x=0.55, y=0.15,
            bgcolor='rgba(255, 255, 255, 0.85)',
            bordercolor='rgba(99, 102, 241, 0.12)',
            font=dict(color='#1E293B', size=9, family="JetBrains Mono")
        ),
        margin=dict(l=50, r=20, t=40, b=40),
        height=280
    )
    return fig


def create_confusion_matrix_heatmap(tp, fp, tn, fn):
    """
    Renders an interactive confusion matrix heatmap in technical indigo/emerald/coral scale.
    """
    z = [[tn, fp], [fn, tp]]
    x = ['Predicted Legitimate', 'Predicted Fraud']
    y = ['Actual Legitimate', 'Actual Fraud']
    
    total = tp + fp + tn + fn
    
    annotations = []
    labels = [
        [f"True Negatives<br><b>{tn:,}</b><br>({(tn/total)*100:.2f}%)", f"False Positives<br><b>{fp:,}</b><br>({(fp/total)*100:.2f}%)"],
        [f"False Negatives<br><b>{fn:,}</b><br>({(fn/total)*100:.2f}%)", f"True Positives<br><b>{tp:,}</b><br>({(tp/total)*100:.2f}%)"]
    ]
    
    for i in range(2):
        for j in range(2):
            is_correct = (i == j)
            # Address contrast bug: small values on light cell background should use dark text (#1E293B)
            cell_value = z[i][j]
            is_dark_cell = (cell_value > total * 0.1)
            
            if is_correct:
                val_color = '#FFFFFF' if is_dark_cell else '#1E293B'
            else:
                val_color = '#EF4444' if i == 1 else '#F59E0B'
            
            annotations.append(dict(
                x=x[j],
                y=y[i],
                text=labels[i][j],
                showarrow=False,
                font=dict(color=val_color, size=11, family="Outfit", weight="bold" if is_correct else "normal")
            ))
            
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[
            [0, 'rgba(99, 102, 241, 0.05)'], 
            [0.1, 'rgba(99, 102, 241, 0.15)'], 
            [1.0, '#4F46E5']
        ],
        showscale=False,
        hoverinfo='none'
    ))
    
    fig.update_layout(
        title=dict(text="<b>Confusion Matrix</b>", font=dict(size=13, color='#1E293B', family='Outfit')),
        annotations=annotations,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color='#1E293B', size=11, family="Outfit")),
        yaxis=dict(tickfont=dict(color='#1E293B', size=11, family="Outfit"), autorange="reversed"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=280
    )
    return fig
