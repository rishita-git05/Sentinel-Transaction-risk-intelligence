"""
Custom CSS styles for SENTINEL Transaction Risk Intelligence.
Implements a refined light-first glassmorphism fintech aesthetic.
"""

def get_custom_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Design Token System */
    :root {
        --bg: #F8FAFC;                 /* Cleaner off-white background */
        --bg-gradient: radial-gradient(circle at 5% 5%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                       radial-gradient(circle at 95% 95%, rgba(13, 148, 136, 0.08) 0%, transparent 40%),
                       #F8FAFC;
        --surface: rgba(255, 255, 255, 0.72); /* Translucent white primary surface (72% opacity) */
        --surface-strong: rgba(255, 255, 255, 0.88); /* Translucent white strong surface (88% opacity) */
        --surface-transparent: transparent;
        --text: #0F172A;               /* Deep charcoal / dark slate text for higher contrast */
        --text-muted: #475569;         /* Slate 600 muted text for better contrast readability */
        --border: rgba(99, 102, 241, 0.15); /* Slightly stronger Indigo-tinted border */
        --border-strong: rgba(99, 102, 241, 0.25);
        --accent: #4F46E5;             /* Deep indigo / violet-blue */
        --accent-secondary: #0D9488;   /* Soft cyan / aqua */
        --accent-decorative: #8B5CF6;  /* Polished Lavender */
        --success: #059669;            /* Polished Emerald green success */
        --warning: #D97706;            /* Polished Amber warning */
        --danger: #EF4444;             /* Coral / Red fraud */
        
        --danger-tint: rgba(239, 68, 68, 0.05);
        --danger-border: rgba(239, 68, 68, 0.15);
        --warning-tint: rgba(217, 119, 6, 0.05);
        --warning-border: rgba(217, 119, 6, 0.15);
        --success-tint: rgba(5, 150, 105, 0.05);
        --success-border: rgba(5, 150, 105, 0.15);
        
        --sidebar-bg-overlay: rgba(255, 255, 255, 0.35);
        --sidebar-border: rgba(99, 102, 241, 0.10);
        --input-bg: rgba(255, 255, 255, 0.85);
        --input-border: rgba(99, 102, 241, 0.18);
        --button-bg: linear-gradient(135deg, #4F46E5, #6366F1);
        --button-text: #FFFFFF;
        --shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.04);
        --shadow-hover: 0 12px 40px 0 rgba(31, 38, 135, 0.06);
        --shadow-button: 0 4px 14px 0 rgba(79, 70, 229, 0.18);
        --shadow-button-hover: 0 6px 20px 0 rgba(79, 70, 229, 0.28);
        --hover-bg: rgba(99, 102, 241, 0.06);
        --active-pill-bg: rgba(99, 102, 241, 0.08);
        --active-pill-border: rgba(99, 102, 241, 0.18);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #090D1A;             /* Deeper midnight navy-black background */
            --bg-gradient: radial-gradient(circle at 5% 5%, rgba(129, 140, 248, 0.15) 0%, transparent 45%),
                           radial-gradient(circle at 95% 95%, rgba(45, 212, 191, 0.10) 0%, transparent 45%),
                           #090D1A;
            --surface: rgba(17, 24, 43, 0.75); /* Rich midnight navy translucent surface */
            --surface-strong: rgba(17, 24, 43, 0.90);
            --text: #F8FAFC;
            --text-muted: #94A3B8;
            --border: rgba(255, 255, 255, 0.12);
            --border-strong: rgba(255, 255, 255, 0.22);
            --accent: #818CF8;
            --accent-secondary: #2DD4BF;
            --accent-decorative: #C084FC;
            --success: #34D399;
            --warning: #FBBF24;
            --danger: #F87171;
            
            --danger-tint: rgba(248, 113, 113, 0.10);
            --danger-border: rgba(248, 113, 113, 0.20);
            --warning-tint: rgba(251, 191, 36, 0.10);
            --warning-border: rgba(251, 191, 36, 0.20);
            --success-tint: rgba(52, 211, 153, 0.10);
            --success-border: rgba(52, 211, 153, 0.20);
            
            --sidebar-bg-overlay: rgba(9, 13, 26, 0.45);
            --sidebar-border: rgba(255, 255, 255, 0.08);
            --input-bg: rgba(9, 13, 26, 0.65);
            --input-border: rgba(255, 255, 255, 0.15);
            --button-bg: linear-gradient(135deg, #6366F1, #818CF8);
            --button-text: #FFFFFF;
            --shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
            --shadow-hover: 0 12px 40px 0 rgba(0, 0, 0, 0.48);
            --shadow-button: 0 4px 14px 0 rgba(0, 0, 0, 0.2);
            --shadow-button-hover: 0 6px 20px 0 rgba(0, 0, 0, 0.3);
            --hover-bg: rgba(255, 255, 255, 0.05);
            --active-pill-bg: rgba(129, 140, 248, 0.12);
            --active-pill-border: rgba(129, 140, 248, 0.22);
        }
    }

    /* Cohesive Global Viewport Background (Solve Leakage) */
    .stApp, 
    div[data-testid="stAppViewContainer"] {
        background: var(--bg-gradient) !important;
        color: var(--text) !important;
        overflow-x: hidden;
    }

    /* Force all nested Streamlit content containers to be transparent and compact padding */
    div[data-testid="stHeader"],
    div[data-testid="stMain"],
    div[data-testid="stMainBlockContainer"],
    .main {
        background: transparent !important;
        background-color: transparent !important;
    }
    div[data-testid="block-container"] {
        background: transparent !important;
        background-color: transparent !important;
        padding-top: 36px !important;
        padding-bottom: 36px !important;
        padding-left: 48px !important;
        padding-right: 48px !important;
    }

    /* Large STATIONARY Blurred Ambient Background Glows */
    .stApp::before {
        content: "";
        position: fixed;
        top: -15vh;
        left: -15vw;
        width: 55vw;
        height: 55vh;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, transparent 70%);
        filter: blur(120px);
        pointer-events: none;
        z-index: 0;
    }
    
    .stApp::after {
        content: "";
        position: fixed;
        bottom: -20vh;
        right: -20vw;
        width: 65vw;
        height: 65vh;
        background: radial-gradient(circle, rgba(13, 148, 136, 0.08) 0%, transparent 70%);
        filter: blur(150px);
        pointer-events: none;
        z-index: 0;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text) !important;
    }

    /* Typography settings - clean Title/Sentence casing */
    h1, h2, h3, h4, h5, h6, 
    [data-testid="stMarkdown"] h1, 
    [data-testid="stMarkdown"] h2, 
    [data-testid="stMarkdown"] h3, 
    [data-testid="stMarkdown"] h4, 
    [data-testid="stMarkdown"] h5, 
    [data-testid="stMarkdown"] h6 {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
        text-transform: none !important;
    }

    p, span, label, li, td, th {
        color: var(--text) !important;
        font-family: 'Outfit', sans-serif;
    }
    
    .mono-val {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Translucent Glass Sidebar Panel */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg-overlay) !important;
        border-right: 1px solid var(--border) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }
    
    /* Hide Streamlit default navigation links */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Form Inputs & Dropdowns as glass system components */
    div[data-baseweb="select"] > div {
        background-color: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:hover {
        background-color: var(--surface-strong) !important;
        border-color: var(--border-strong) !important;
    }
    
    /* Ensure selection text is always readable */
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span {
        color: var(--text) !important;
    }
    
    /* Dropdown popover list */
    div[role="listbox"] {
        background-color: var(--sidebar-bg-overlay) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 8px !important;
    }
    div[role="listbox"] div[role="option"] {
        color: var(--text) !important;
        transition: all 0.15s ease !important;
    }
    div[role="listbox"] div[role="option"]:hover {
        background-color: var(--hover-bg) !important;
        color: var(--text) !important;
    }
    div[role="listbox"] div[role="option"][aria-selected="true"] {
        background-color: var(--active-pill-bg) !important;
        color: var(--accent) !important;
        font-weight: 600 !important;
    }

    /* Primary CTA buttons */
    .stButton>button {
        background: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        box-shadow: var(--shadow-button) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: none !important;
        letter-spacing: 0px !important;
        width: auto !important;
        display: inline-block !important;
    }
    .stButton>button:hover {
        box-shadow: var(--shadow-button-hover) !important;
        transform: translateY(-1px) !important;
        color: var(--button-text) !important;
        opacity: 0.95 !important;
    }

    /* Secondary actions / Glass Outline Buttons */
    .neutral-btn-wrapper .stButton > button {
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
        box-shadow: none !important;
    }
    .neutral-btn-wrapper .stButton > button:hover {
        background: var(--surface-strong) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        transform: none !important;
    }

    /* Sidebar Navigation Links - Translucent & Transparent Inactive States */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: var(--text-muted) !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 16px !important;
        margin-bottom: 4px !important;
        box-shadow: none !important;
        width: 100% !important;
        text-transform: none !important;
        letter-spacing: 0px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--hover-bg) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    /* Active Sidebar pill override */
    .active-nav-wrapper .stButton > button {
        background: var(--active-pill-bg) !important;
        color: var(--accent) !important;
        font-weight: 600 !important;
        border: 1px solid var(--active-pill-border) !important;
    }

    /* Timeline flow design */
    .timeline-pipeline {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        position: relative;
        margin: 30px 0;
        padding-bottom: 20px;
        border-bottom: 1px solid var(--border);
    }
    .timeline-line {
        position: absolute;
        top: 15px;
        left: 8%;
        right: 8%;
        height: 2px;
        background: var(--border);
        z-index: 1;
    }
    .timeline-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
        z-index: 2;
        flex: 1;
    }
    .timeline-circle {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--bg);
        border: 2px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 700;
        color: var(--text-muted);
        transition: all 0.3s ease;
    }
    .timeline-step.active .timeline-circle {
        background: var(--accent);
        border-color: var(--accent);
        color: #FFFFFF;
        box-shadow: 0 0 12px rgba(79, 70, 229, 0.2);
    }
    .timeline-title {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 700;
        color: var(--text);
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .timeline-desc {
        font-family: 'Outfit', sans-serif;
        font-size: 10px;
        color: var(--text-muted);
        text-align: center;
        margin-top: 2px;
        max-width: 140px;
    }

    /* Section Headers */
    .question-header {
        margin-top: 24px;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }
    .question-title {
        font-family: 'Outfit', sans-serif;
        font-size: 16px;
        font-weight: 600;
        color: var(--text);
        margin: 0;
        text-transform: none;
        letter-spacing: 0px;
    }

    /* Glass Cards */
    .glass-card,
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: var(--shadow) !important;
        margin-bottom: 24px !important;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .glass-card:hover,
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-hover) !important;
    }

    /* KPI Cards inside Glass Cards */
    .kpi-card {
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        min-height: 130px !important;
        box-sizing: border-box !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: var(--shadow) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .kpi-card.kpi-highlight {
        border: 2px solid var(--border-strong) !important;
        border-top: 4px solid var(--accent) !important;
        background: var(--surface-strong) !important;
    }
    .kpi-card.kpi-highlight .kpi-value {
        color: var(--accent) !important;
    }
    .kpi-card.kpi-info {
        border-left: 3.5px solid var(--accent) !important;
        background: var(--surface-strong) !important;
    }
    .kpi-card.kpi-success {
        border-left: 3.5px solid var(--success) !important;
        background: var(--success-tint) !important;
        border-color: var(--success-border) !important;
    }
    .kpi-card.kpi-success .kpi-value {
        color: var(--success) !important;
    }
    .kpi-card.kpi-warning {
        border-left: 3.5px solid var(--warning) !important;
        background: var(--warning-tint) !important;
        border-color: var(--warning-border) !important;
    }
    .kpi-card.kpi-warning .kpi-value {
        color: var(--warning) !important;
    }
    .kpi-card.kpi-danger {
        border-left: 3.5px solid var(--danger) !important;
        background: var(--danger-tint) !important;
        border-color: var(--danger-border) !important;
    }
    .kpi-card.kpi-danger .kpi-value {
        color: var(--danger) !important;
    }
    .kpi-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-hover) !important;
        border-color: var(--border-strong) !important;
    }
    .kpi-label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        font-weight: 600 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 6px !important;
        border-bottom: none !important;
        padding-bottom: 0px !important;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        color: var(--text) !important;
    }
    .kpi-sub {
        font-family: 'Outfit', sans-serif !important;
        font-size: 11.5px !important;
        color: var(--text-muted) !important;
        margin-top: 4px !important;
    }

    /* Recommended Action Box */
    .action-container {
        padding: 16px 20px;
        border-radius: 12px;
        margin-top: 20px;
    }

    /* Flow step descriptions */
    .flow-step-desc, .kpi-sub, .section-subtitle, .timeline-desc {
        color: var(--text-muted) !important;
    }
    .kpi-label, .tech-detail-label {
        color: var(--text-muted) !important;
    }

    /* Streamlit expander overrides */
    .streamlit-expanderHeader {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-weight: 500 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    .streamlit-expanderContent {
        background-color: var(--surface-strong) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 16px !important;
    }

    /* Streamlit tabs overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 0px !important;
        color: var(--text-muted) !important;
        border: 1px solid transparent !important;
        padding: 8px 16px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        text-transform: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--hover-bg) !important;
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
        font-weight: 600 !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] [role="slider"] {
        background-color: var(--accent) !important;
        border: 2px solid #FFFFFF !important;
    }
    div[data-testid="stSlider"] div[role="presentation"] {
        background-color: var(--border-strong) !important;
    }

    /* Dataframe layout overrides */
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        background-color: var(--surface) !important;
    }
    
    hr {
        border: 0;
        border-top: 1px solid var(--border) !important;
        margin: 24px 0 !important;
    }

    /* Custom styles for segment segmented controls (radio buttons wrapped as segments) */
    div[data-testid="stRadio"] label:not([data-testid="stWidgetLabel"]) div div > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] label:not([data-testid="stWidgetLabel"]) div div > div:nth-child(2) {
        margin-left: 0px !important;
        padding-left: 0px !important;
    }
    div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    div[data-testid="stRadio"] label:not([data-testid="stWidgetLabel"]) {
        background-color: transparent !important;
        border-radius: 16px !important;
        padding: 6px 16px !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-size: 12.5px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] label:not([data-testid="stWidgetLabel"]):hover:not(:has(input[type="radio"]:checked)) {
        background-color: var(--hover-bg) !important;
        color: var(--text) !important;
    }
    div[data-testid="stRadio"] label:not([data-testid="stWidgetLabel"]):has(input[type="radio"]:checked) {
        background-color: var(--accent) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        box-shadow: var(--shadow-button) !important;
    }

    /* Sidebar Vertical Radio Navigation Overrides */
    [data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        flex-direction: column !important;
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        gap: 4px !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:not([data-testid="stWidgetLabel"]) {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        background: transparent !important;
        color: var(--text-muted) !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
        padding: 10px 16px !important;
        margin: 0 !important;
        cursor: pointer !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:not([data-testid="stWidgetLabel"]):hover:not(:has(input[type="radio"]:checked)) {
        background: var(--hover-bg) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:not([data-testid="stWidgetLabel"]):has(input[type="radio"]:checked) {
        background: var(--active-pill-bg) !important;
        color: var(--accent) !important;
        font-weight: 600 !important;
        border: 1px solid var(--active-pill-border) !important;
        box-shadow: none !important;
    }

    /* Hide the radio widget label text blocks globally and collapse spacing */
    div[data-testid="stRadio"] > label,
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
        display: none !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        padding-top: 18px !important;
        padding-bottom: 18px !important;
    }
    </style>
    """
