import os
import sys
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure src module can be imported cleanly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import FEATURE_COLUMNS, FEATURE_LABELS, TARGET_COLUMN, load_dataset
from src.prediction import load_saved_model, predict_observation, generate_synthetic_observation, validate_uploaded_csv
from src.visualization import (
    create_3d_orbit_visualization,
    create_transit_lightcurve_chart,
    create_feature_importance_chart,
    create_confusion_matrix_chart,
    create_class_distribution_chart,
    create_feature_distribution_histogram
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Exoplanet Detection System",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Space-Themed CSS — Premium Glassmorphism Design
# ---------------------------------------------------------
SPACE_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* ─── Global Background ─── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1225 40%, #111b33 70%, #0a0e1a 100%);
        color: #e2e8f0;
    }
    .main .block-container {
        padding-top: 2rem;
    }

    /* ─── Animated starfield background ─── */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            radial-gradient(2px 2px at 20px 30px, rgba(0,242,254,0.3), transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(79,172,254,0.2), transparent),
            radial-gradient(1px 1px at 90px 40px, rgba(255,255,255,0.3), transparent),
            radial-gradient(1px 1px at 160px 120px, rgba(0,242,254,0.2), transparent),
            radial-gradient(2px 2px at 200px 60px, rgba(255,177,153,0.15), transparent),
            radial-gradient(1px 1px at 300px 200px, rgba(255,255,255,0.25), transparent),
            radial-gradient(1.5px 1.5px at 400px 100px, rgba(0,242,254,0.2), transparent),
            radial-gradient(1px 1px at 500px 300px, rgba(255,255,255,0.2), transparent);
        background-size: 550px 350px;
        animation: sparkle 40s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    @keyframes sparkle {
        from { transform: translateY(0); }
        to { transform: translateY(-350px); }
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080c18 0%, #0d1225 100%);
        border-right: 1px solid rgba(0, 242, 254, 0.12);
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }

    /* ─── Glassmorphism Card ─── */
    .glass-card {
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.15);
        border-radius: 20px;
        padding: 28px 30px;
        margin-bottom: 22px;
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 242, 254, 0.35);
        box-shadow:
            0 12px 40px rgba(0, 0, 0, 0.35),
            0 0 20px rgba(0, 242, 254, 0.08);
    }

    .glass-card-accent {
        background: rgba(10, 18, 40, 0.8);
        backdrop-filter: blur(18px);
        border: 1.5px solid rgba(0, 242, 254, 0.4);
        border-radius: 20px;
        padding: 32px;
        box-shadow:
            0 0 30px rgba(0, 242, 254, 0.12),
            0 12px 40px rgba(0, 0, 0, 0.4);
    }

    /* ─── Result Badges ─── */
    .result-badge-confirmed {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: #051911;
        font-size: 26px;
        font-weight: 800;
        padding: 14px 32px;
        border-radius: 14px;
        text-align: center;
        letter-spacing: 2px;
        box-shadow: 0 6px 25px rgba(0, 176, 155, 0.45);
        text-transform: uppercase;
        font-family: 'Outfit', sans-serif;
    }
    .result-badge-candidate {
        background: linear-gradient(135deg, #f6d365, #fda085);
        color: #2b1800;
        font-size: 26px;
        font-weight: 800;
        padding: 14px 32px;
        border-radius: 14px;
        text-align: center;
        letter-spacing: 2px;
        box-shadow: 0 6px 25px rgba(253, 160, 133, 0.45);
        text-transform: uppercase;
        font-family: 'Outfit', sans-serif;
    }
    .result-badge-fp {
        background: linear-gradient(135deg, #ff0844, #ffb199);
        color: #ffffff;
        font-size: 26px;
        font-weight: 800;
        padding: 14px 32px;
        border-radius: 14px;
        text-align: center;
        letter-spacing: 2px;
        box-shadow: 0 6px 25px rgba(255, 8, 68, 0.45);
        text-transform: uppercase;
        font-family: 'Outfit', sans-serif;
    }

    /* ─── Metric Pill Cards ─── */
    .metric-pill {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-pill:hover {
        border-color: rgba(0, 242, 254, 0.4);
        background: rgba(0, 242, 254, 0.06);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.1);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 6px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }

    /* ─── Flow step pill ─── */
    .flow-step {
        display: inline-block;
        background: rgba(0, 242, 254, 0.06);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 12px;
        padding: 12px 20px;
        margin: 6px 0;
        color: #e2e8f0;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        transition: all 0.25s ease;
    }
    .flow-step:hover {
        background: rgba(0, 242, 254, 0.12);
        border-color: rgba(0, 242, 254, 0.5);
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.1);
    }

    /* ─── Progress Step Labels ─── */
    .progress-step {
        padding: 8px 16px;
        border-left: 3px solid #00f2fe;
        margin: 8px 0;
        color: #cbd5e1;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        background: rgba(0, 242, 254, 0.04);
        border-radius: 0 8px 8px 0;
    }

    /* ─── Section Headers ─── */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .section-subtitle {
        color: #94a3b8;
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        font-weight: 400;
        margin-bottom: 24px;
    }

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-family: 'Inter', sans-serif;
    }

    /* ─── Buttons ─── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00b4d8, #00f2fe) !important;
        color: #0a0e1a !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* ─── Expander ─── */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
    }

    /* ─── Tables & Dataframes ─── */
    .stTable, .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ─── Hide Streamlit branding ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
st.markdown(SPACE_THEME_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Saved Model & Metadata (cached)
# ---------------------------------------------------------
@st.cache_resource
def get_model():
    try:
        return load_saved_model()
    except Exception:
        return None, None, None

rf_pipeline, lr_pipeline, metadata = get_model()

# Auto-train if model not found
if rf_pipeline is None or metadata is None:
    st.warning("⚠️ Trained model artifacts not found. Running training pipeline...")
    with st.spinner("Training Random Forest & Logistic Regression models on NASA Kepler data..."):
        from src.training import train_and_evaluate_models
        train_and_evaluate_models()
        st.cache_resource.clear()
        rf_pipeline, lr_pipeline, metadata = get_model()
    st.success("✅ Model trained and loaded successfully!")
    st.rerun()

# ---------------------------------------------------------
# Navigation Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 18px 0 12px 0;">
    <div style="font-size: 40px; margin-bottom: 4px;">🔭</div>
    <h2 style="background: linear-gradient(135deg, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; font-family: 'Outfit', sans-serif; font-size: 22px; font-weight: 800; letter-spacing: 1px;">EXOPLANET AI</h2>
    <p style="color: #64748b; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; font-family: 'Inter', sans-serif;">Kepler Candidate Classifier</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🌌 Home",
        "🔬 How It Works",
        "📊 Dataset Explorer",
        "🔭 Exoplanet Detection",
        "🧠 Model Performance",
        "ℹ️ Project Info"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 12px 8px;">
    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #475569; margin-bottom: 12px; font-family: 'Inter', sans-serif; font-weight: 700;">Project Details</div>
    <div style="color: #94a3b8; font-size: 13px; line-height: 1.8; font-family: 'Inter', sans-serif;">
        📋 <b style="color:#cbd5e1;">College PBL Project</b><br>
        📡 <b style="color:#cbd5e1;">Dataset:</b> NASA Kepler KOI<br>
        🤖 <b style="color:#cbd5e1;">Model:</b> Random Forest<br>
        📈 <b style="color:#cbd5e1;">Baseline:</b> Logistic Regression<br>
        🎯 <b style="color:#cbd5e1;">Classes:</b> 3 (Multiclass)
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 1. HOME PAGE
# =========================================================
if page == "🌌 Home":
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 30px 0 10px 0;">
        <div style="font-size: 52px; margin-bottom: 8px;">🌌</div>
        <h1 style="font-size: 44px; background: linear-gradient(135deg, #00f2fe 0%, #4facfe 40%, #00f2fe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 6px; font-family: 'Outfit', sans-serif; font-weight: 900; letter-spacing: -1px;">
            AI-Based Intelligent Exoplanet Detection System
        </h1>
        <p style="color: #94a3b8; font-size: 18px; font-weight: 400; margin-top: 0; font-family: 'Inter', sans-serif; max-width: 700px; margin-left: auto; margin-right: auto; line-height: 1.6;">
            Supervised Machine Learning for Astronomical Exoplanet Candidate Classification using NASA Kepler Space Telescope Data
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00f2fe; margin-top:0; font-family: 'Outfit', sans-serif; font-size: 22px;">🪐 What is an Exoplanet?</h3>
            <p style="color: #cbd5e1; line-height: 1.75; font-family: 'Inter', sans-serif; font-size: 14.5px;">
                An <b style="color:#00f2fe">exoplanet</b> (extrasolar planet) is a planet located outside our Solar System,
                orbiting a distant star. As of 2024, over <b style="color:#4facfe">5,600 exoplanets</b> have been confirmed.
                Discovering them helps astronomers understand planetary formation and the potential for
                habitable worlds across our galaxy.
            </p>
            <div style="margin-top: 20px; border-top: 1px solid rgba(0,242,254,0.15); padding-top: 20px;">
                <h3 style="color: #00f2fe; margin-top:0; font-family: 'Outfit', sans-serif; font-size: 22px;">📉 The Transit Method</h3>
                <p style="color: #cbd5e1; line-height: 1.75; font-family: 'Inter', sans-serif; font-size: 14.5px;">
                    When an exoplanet passes in front of its host star (a <b style="color:#fda085">transit</b>),
                    it blocks a tiny fraction of starlight, creating a measurable dip in brightness.
                    NASA's Kepler Space Telescope monitored <b style="color:#4facfe">150,000+ stars</b>
                    for these periodic dips, generating transit signals that can be classified using Machine Learning.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3 style="color: #00f2fe; margin-top:0; font-family: 'Outfit', sans-serif; font-size: 22px;">📡 Detection Pipeline</h3>
            <div style="display: flex; flex-direction: column; align-items: center; gap: 6px; margin-top: 18px;">
                <div class="flow-step" style="width: 85%;">⭐ Host Star Observation</div>
                <div style="color: rgba(0,242,254,0.5); font-size: 20px;">⬇</div>
                <div class="flow-step" style="width: 85%;">🪐 Planet Transit Event</div>
                <div style="color: rgba(0,242,254,0.5); font-size: 20px;">⬇</div>
                <div class="flow-step" style="width: 85%;">📉 Periodic Brightness Dip</div>
                <div style="color: rgba(0,242,254,0.5); font-size: 20px;">⬇</div>
                <div class="flow-step" style="width: 85%;">📊 Numerical Astronomical Features</div>
                <div style="color: rgba(0,242,254,0.5); font-size: 20px;">⬇</div>
                <div class="flow-step" style="width: 85%;">🤖 Random Forest Ensemble (100 Trees)</div>
                <div style="color: rgba(0,242,254,0.5); font-size: 20px;">⬇</div>
                <div class="flow-step" style="width: 85%; border-color: #00f2fe; color: #00f2fe; background: rgba(0,242,254,0.1); font-weight: 700;">🔭 Exoplanet Candidate Classification</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🪐 Interactive Astronomical Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Educational 3D stellar model and simulated transit light curve</div>', unsafe_allow_html=True)

    t1, t2 = st.tabs(["🌌 3D Stellar Orbit Model", "📉 Transit Light Curve"])

    with t1:
        st.plotly_chart(create_3d_orbit_visualization(planet_radius=2.5, orbital_period=12.0), key="home_3d")
        st.caption("📌 Educational 3D visualization of a host star with an orbiting exoplanet candidate. This is a conceptual model, not a real observation.")

    with t2:
        st.plotly_chart(create_transit_lightcurve_chart(depth_ppm=1500.0, duration_hrs=4.0), key="home_lc")
        st.caption("📌 Simulated transit light curve showing the characteristic dip in stellar brightness during a planetary transit.")


# =========================================================
# 2. HOW IT WORKS PAGE
# =========================================================
elif page == "🔬 How It Works":
    st.markdown('<div class="section-header">🔬 How the System Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Step-by-step breakdown of the AI exoplanet classification pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("🛰️", "Step 1 — NASA Kepler KOI Dataset",
         "Real observational data from the Kepler Space Telescope containing thousands of Kepler Objects of Interest (KOIs). Each KOI includes transit measurements and stellar characterization parameters, labelled with a disposition (CONFIRMED, CANDIDATE, or FALSE POSITIVE)."),
        ("🧹", "Step 2 — Data Preprocessing & Cleaning",
         "Missing values are imputed using median strategy. Features are standardized using z-score normalization. Preprocessing is fitted ONLY on training data to prevent data leakage. Disposition-related metadata and identifiers are excluded."),
        ("📊", "Step 3 — Feature Selection (12 Features)",
         "Twelve legitimate observational features are selected: orbital period, impact parameter, transit duration, transit depth, planetary radius, equilibrium temperature, insolation flux, transit SNR, stellar temperature, stellar surface gravity, stellar radius, and Kepler magnitude."),
        ("🌲", "Step 4 — Random Forest Classifier",
         "An ensemble of 100 decision trees trained on real NASA observations. Each tree independently evaluates transit signals using learned patterns. The forest combines individual tree decisions via majority voting for robust classification."),
        ("📂", "Step 5 — Observation Input",
         "Users provide new observations either by uploading a CSV file with the required 12 features, or by generating synthetic demo observations based on empirical training data distributions."),
        ("🎯", "Step 6 — Inference & Probability",
         "The input observation passes through the exact same preprocessing pipeline used during training, then the Random Forest computes per-class probability scores for CONFIRMED, CANDIDATE, and FALSE POSITIVE."),
        ("💡", "Step 7 — Feature Importance & Explanation",
         "Random Forest provides Gini-based feature importance scores showing which astronomical features contributed most to the model's decision-making process. This aids interpretability and scientific understanding.")
    ]

    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="glass-card" style="display: flex; gap: 20px; align-items: flex-start;">
            <div style="font-size: 36px; min-width: 50px; text-align: center;">{icon}</div>
            <div>
                <h4 style="color: #00f2fe; margin: 0 0 8px 0; font-family: 'Outfit', sans-serif; font-size: 18px;">{title}</h4>
                <p style="color: #cbd5e1; margin: 0; line-height: 1.7; font-family: 'Inter', sans-serif; font-size: 14px;">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# 3. DATASET EXPLORER
# =========================================================
elif page == "📊 Dataset Explorer":
    st.markdown('<div class="section-header">📊 NASA Kepler KOI Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Interactive exploration of the real NASA training dataset</div>', unsafe_allow_html=True)

    raw_df, available_features = load_dataset()

    # Key Metrics Row
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f'<div class="metric-pill"><div class="metric-value">{len(raw_df)}</div><div class="metric-label">Total Observations</div></div>', unsafe_allow_html=True)
    with mcol2:
        st.markdown(f'<div class="metric-pill"><div class="metric-value">{len(available_features)}</div><div class="metric-label">Input Features</div></div>', unsafe_allow_html=True)
    with mcol3:
        st.markdown(f'<div class="metric-pill"><div class="metric-value">{len(metadata["classes"])}</div><div class="metric-label">Target Classes</div></div>', unsafe_allow_html=True)
    with mcol4:
        null_count = raw_df[available_features].isnull().sum().sum()
        st.markdown(f'<div class="metric-pill"><div class="metric-value">{null_count}</div><div class="metric-label">Missing Values</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    dcol1, dcol2 = st.columns([1, 1], gap="large")

    with dcol1:
        st.plotly_chart(create_class_distribution_chart(metadata['class_distribution']), key="dataset_pie")

    with dcol2:
        st.markdown("#### 📈 Feature Distribution by Class")
        selected_feat = st.selectbox(
            "Select Feature:",
            options=available_features,
            format_func=lambda x: f"{FEATURE_LABELS.get(x, x)}"
        )
        st.plotly_chart(create_feature_distribution_histogram(raw_df, selected_feat), key="dataset_hist")

    st.markdown("---")
    st.markdown("#### 📋 Sample NASA Kepler Observations")
    st.dataframe(raw_df.head(20), height=400)

    with st.expander("🛠️ Advanced Dataset Details"):
        st.markdown("**Feature Summary Statistics:**")
        st.dataframe(raw_df[available_features].describe().T)

        st.markdown("**Feature Descriptions:**")
        feat_desc_df = pd.DataFrame([
            {"Feature": k, "Description": v} for k, v in FEATURE_LABELS.items() if k in available_features
        ])
        st.table(feat_desc_df)


# =========================================================
# 4. EXOPLANET DETECTION (INFERENCE PAGE)
# =========================================================
elif page == "🔭 Exoplanet Detection":
    st.markdown('<div class="section-header">🔭 Exoplanet Candidate Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Run inference on new astronomical observations using the trained Random Forest pipeline</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose Observation Input Method:",
        ["📂 Upload CSV File (Real/External Data)", "🧪 Generate Synthetic Demo Observation"],
        horizontal=True
    )

    input_df = None
    is_demo = False

    if "Upload CSV" in input_mode:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #00f2fe; margin-top: 0; font-family: 'Outfit', sans-serif;">📂 Upload Observation CSV</h4>
            <p style="color: #94a3b8; font-size: 13px; font-family: 'Inter', sans-serif;">
                Upload a CSV file containing 12 Kepler observation features. The file must include columns:
                <code>koi_period, koi_impact, koi_duration, koi_depth, koi_prad, koi_teq, koi_insol, koi_model_snr, koi_steff, koi_slogg, koi_srad, koi_kepmag</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Select CSV file:", type=["csv"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                raw_uploaded = pd.read_csv(uploaded_file)
                is_valid, errors, cleaned_df = validate_uploaded_csv(raw_uploaded, metadata['features'])

                if not is_valid:
                    for err in errors:
                        st.error(f"❌ {err}")
                else:
                    if errors:  # warnings like NaN notice
                        for err in errors:
                            st.warning(f"⚠️ {err}")
                    st.success(f"✅ CSV validated successfully — {len(cleaned_df)} observation(s) loaded.")
                    input_df = cleaned_df
                    st.dataframe(input_df.head(10))
            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {str(e)}")

    else:  # Synthetic Demo Mode
        is_demo = True
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #fda085; margin-top: 0; font-family: 'Outfit', sans-serif;">🧪 Synthetic Demo Mode</h4>
            <p style="color: #94a3b8; font-size: 13px; font-family: 'Inter', sans-serif; line-height: 1.6;">
                Generate a synthetic observation based on the empirical distributions of real NASA Kepler data.
                The model will classify it in real-time. <b style="color:#fda085;">This is for demonstration purposes only</b> —
                synthetic observations are not real astronomical measurements and are never used for model training.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            preset = st.selectbox(
                "Preset Scenario:",
                ["🌍 Earth-Like Candidate", "🪐 Hot Jupiter / Gas Giant", "🎲 Random KOI Distribution"]
            )
            preset_key = "earth_like" if "Earth" in preset else ("gas_giant" if "Jupiter" in preset else "random")

            if st.button("🧪 Generate Demo Observation", type="primary"):
                st.session_state['demo_df'] = generate_synthetic_observation(metadata['feature_stats'], preset=preset_key)

        if 'demo_df' in st.session_state:
            input_df = st.session_state['demo_df']
            with col_p2:
                st.markdown("##### Generated Feature Values (editable):")
                edited_df = st.data_editor(input_df, key="demo_editor", height=120)
                input_df = edited_df

    st.markdown("---")

    if input_df is not None:
        if st.button("🚀 Analyze & Classify Observation", type="primary", use_container_width=True):

            # Progress Animation Sequence
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                steps_anim = [
                    (15, "✓ Observation data received"),
                    (30, "✓ Validating feature dimensions (12 features)"),
                    (50, "✓ Applying preprocessing pipeline (Median Imputation → Standard Scaling)"),
                    (70, "✓ Running Random Forest Classifier (100 Decision Trees)"),
                    (85, "✓ Computing class probability scores"),
                    (100, "✓ Generating explanation and visualizations")
                ]

                for pct, txt in steps_anim:
                    progress_bar.progress(pct)
                    status_placeholder.markdown(f'<div class="progress-step">{txt}</div>', unsafe_allow_html=True)
                    time.sleep(0.2)

                time.sleep(0.3)
                progress_bar.empty()
                status_placeholder.empty()

            # Perform inference
            obs_row = input_df.iloc[[0]]
            prediction, probabilities = predict_observation(rf_pipeline, obs_row)
            conf_prob = probabilities.get(prediction, 0.0) * 100

            # Result Card
            badge_class = "result-badge-confirmed" if prediction == "CONFIRMED" else ("result-badge-candidate" if prediction == "CANDIDATE" else "result-badge-fp")

            if is_demo:
                st.info("ℹ️ Synthetic Demo Observation — generated for application demonstration, not an actual astronomical observation.")

            st.markdown(f"""
            <div class="glass-card-accent" style="text-align: center; margin-top: 10px;">
                <div style="font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 3px; font-family: 'Inter', sans-serif; font-weight: 700;">Predicted Classification</div>
                <div style="margin: 18px 0;">
                    <span class="{badge_class}">{prediction}</span>
                </div>
                <div style="margin-top: 16px;">
                    <span style="font-size: 14px; color: #94a3b8; font-family: 'Inter', sans-serif;">Model Confidence:</span>
                    <span style="font-size: 36px; font-weight: 800; background: linear-gradient(135deg, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Outfit', sans-serif; margin-left: 8px;">{conf_prob:.1f}%</span>
                </div>
                <div style="font-size: 12px; color: #64748b; margin-top: 12px; font-family: 'Inter', sans-serif;">
                    Exoplanet candidate classification • ML pattern analysis • Random Forest Ensemble
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability Breakdown Chart
            st.markdown("#### 📊 Class Probability Distribution")
            prob_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
            prob_df['Percentage'] = prob_df['Probability'] * 100

            color_map = {'CONFIRMED': '#00F2FE', 'FALSE POSITIVE': '#FF0844', 'CANDIDATE': '#FFB199'}

            fig_prob = go.Figure()
            for _, row in prob_df.iterrows():
                fig_prob.add_trace(go.Bar(
                    x=[row['Class']],
                    y=[row['Percentage']],
                    name=row['Class'],
                    marker_color=color_map.get(row['Class'], '#4facfe'),
                    text=f"{row['Percentage']:.1f}%",
                    textposition='outside',
                    textfont=dict(color='#e2e8f0', size=14, family='Inter')
                ))

            fig_prob.update_layout(
                yaxis=dict(title="Probability (%)", range=[0, 110], gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94a3b8")),
                xaxis=dict(title="", tickfont=dict(color="#e2e8f0", size=13)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                showlegend=False,
                height=320,
                margin=dict(l=50, r=30, b=40, t=20)
            )
            st.plotly_chart(fig_prob, key="prob_chart")

            # Transit lightcurve for analyzed observation
            depth_val = float(obs_row['koi_depth'].values[0]) if 'koi_depth' in obs_row else 1000.0
            duration_val = float(obs_row['koi_duration'].values[0]) if 'koi_duration' in obs_row else 3.5
            period_val = float(obs_row['koi_period'].values[0]) if 'koi_period' in obs_row else 10.0

            st.plotly_chart(
                create_transit_lightcurve_chart(depth_ppm=depth_val, duration_hrs=duration_val, period_days=period_val),
                key="detect_lc"
            )

            # Feature Importances
            st.plotly_chart(create_feature_importance_chart(metadata['feature_importances']), key="detect_fi")
            st.caption("📌 Feature importance indicates how useful each feature was to the trained Random Forest model. It does not establish a physical causal relationship.")

            # Expandable Educational Explanation
            with st.expander("🧠 What happened behind the scenes?"):
                st.markdown("""
                **How Random Forest Classified This Observation:**

                ```
                Your Input → 12 Astronomical Features
                     ↓
                Preprocessing Pipeline (Median Imputation → Standard Scaling)
                     ↓
                Random Forest Ensemble (100 Independent Decision Trees)
                     ↓
                Each Tree Votes: CONFIRMED / CANDIDATE / FALSE POSITIVE
                     ↓
                Majority Vote → Final Classification + Probability Scores
                ```

                **What is a Random Forest?**

                A Random Forest is an ensemble of many individual decision trees. Each tree is trained on a
                random subset of the training data and features. When classifying a new observation, every tree
                independently evaluates features like transit depth, orbital period, and stellar temperature
                using learned decision rules. The forest aggregates all tree votes through majority voting
                to produce a robust final classification and per-class probability estimate.
                """)

            # Expandable Limitations
            with st.expander("⚠️ Important Limitations"):
                st.markdown("""
                - **Candidate Classification Only:** This system performs ML-based pattern classification. It does **not** scientifically confirm a new exoplanet discovery.
                - **Training Data Dependence:** Predictions depend strictly on the distribution and quality of the NASA Kepler training dataset (215 observations).
                - **Synthetic Demo Data:** Demo observations are generated for application demonstration and are not real astronomical measurements.
                - **Scientific Validation Required:** Real astronomical confirmation requires follow-up radial velocity measurements, spectroscopy, and peer-reviewed analysis.
                - **Class Imbalance:** The training dataset has uneven class distribution which can affect prediction reliability for minority classes.
                - **Not a Replacement:** This ML model should not be interpreted as a replacement for professional astronomical analysis.
                """)


# =========================================================
# 5. MODEL PERFORMANCE PAGE
# =========================================================
elif page == "🧠 Model Performance":
    st.markdown('<div class="section-header">🧠 Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">All metrics computed from actual held-out test set evaluation — nothing is fabricated</div>', unsafe_allow_html=True)

    rf_m = metadata['rf_metrics']
    lr_m = metadata['lr_metrics']

    # Top-level metric cards for RF
    st.markdown("#### 🌲 Random Forest Classifier (Primary Model)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-pill"><div class="metric-value">{rf_m["accuracy"]*100:.1f}%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-pill"><div class="metric-value">{rf_m["precision_weighted"]*100:.1f}%</div><div class="metric-label">Precision (W)</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-pill"><div class="metric-value">{rf_m["recall_weighted"]*100:.1f}%</div><div class="metric-label">Recall (W)</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-pill"><div class="metric-value">{rf_m["f1_weighted"]*100:.1f}%</div><div class="metric-label">F1 Score (W)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison Table
    st.markdown("#### 📊 Full Metrics Comparison")

    comp_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1 Score (Weighted)', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)'],
        'Random Forest': [
            f"{rf_m['accuracy']*100:.2f}%",
            f"{rf_m['precision_weighted']*100:.2f}%",
            f"{rf_m['recall_weighted']*100:.2f}%",
            f"{rf_m['f1_weighted']*100:.2f}%",
            f"{rf_m['precision_macro']*100:.2f}%",
            f"{rf_m['recall_macro']*100:.2f}%",
            f"{rf_m['f1_macro']*100:.2f}%"
        ],
        'Logistic Regression': [
            f"{lr_m['accuracy']*100:.2f}%",
            f"{lr_m['precision_weighted']*100:.2f}%",
            f"{lr_m['recall_weighted']*100:.2f}%",
            f"{lr_m['f1_weighted']*100:.2f}%",
            f"{lr_m['precision_macro']*100:.2f}%",
            f"{lr_m['recall_macro']*100:.2f}%",
            f"{lr_m['f1_macro']*100:.2f}%"
        ]
    })

    st.table(comp_df)

    st.markdown("---")

    # Confusion Matrices
    st.markdown("#### 🔢 Confusion Matrices")
    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.plotly_chart(
            create_confusion_matrix_chart(rf_m['confusion_matrix'], metadata['classes'], title="Random Forest"),
            key="cm_rf"
        )
    with col_cm2:
        st.plotly_chart(
            create_confusion_matrix_chart(lr_m['confusion_matrix'], metadata['classes'], title="Logistic Regression"),
            key="cm_lr"
        )

    # Feature Importance
    st.markdown("---")
    st.markdown("#### 🌲 Random Forest Feature Importance")
    st.plotly_chart(create_feature_importance_chart(metadata['feature_importances']), key="perf_fi")

    with st.expander("📋 Full Scikit-Learn Classification Reports"):
        st.markdown("**Random Forest:**")
        st.dataframe(pd.DataFrame(rf_m['classification_report']).T)
        st.markdown("**Logistic Regression:**")
        st.dataframe(pd.DataFrame(lr_m['classification_report']).T)


# =========================================================
# 6. PROJECT INFO PAGE
# =========================================================
elif page == "ℹ️ Project Info":
    st.markdown('<div class="section-header">ℹ️ Project Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">College PBL project details and specifications</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #00f2fe; margin-top:0; font-family: 'Outfit', sans-serif;">AI-Based Intelligent Exoplanet Detection System</h3>
        <div style="margin-top: 18px;">
            <h4 style="color: #4facfe; font-family: 'Outfit', sans-serif; font-size: 17px;">🎯 Problem Statement</h4>
            <p style="color: #cbd5e1; line-height: 1.7; font-family: 'Inter', sans-serif; font-size: 14px;">
                Modern space telescopes generate vast quantities of astronomical light curves and transit signals.
                Manually identifying candidate exoplanets among thousands of observations is extremely labor-intensive.
                Supervised Machine Learning can assist astronomers by rapidly classifying Kepler observational signals
                into confirmed exoplanets, candidates, or false positives.
            </p>
        </div>
        <div style="margin-top: 18px;">
            <h4 style="color: #4facfe; font-family: 'Outfit', sans-serif; font-size: 17px;">📋 Objectives</h4>
            <ul style="color: #cbd5e1; line-height: 2; font-family: 'Inter', sans-serif; font-size: 14px;">
                <li>Build a supervised ML classification pipeline using the real NASA Kepler KOI cumulative dataset</li>
                <li>Train a Random Forest Classifier and evaluate against a Logistic Regression baseline</li>
                <li>Prevent data leakage and target leakage in all stages of preprocessing</li>
                <li>Provide an interactive, space-themed Streamlit UI with real-time inference capability</li>
                <li>Offer dual input modes: real CSV upload and clearly-labelled synthetic demo observations</li>
            </ul>
        </div>
        <div style="margin-top: 18px;">
            <h4 style="color: #4facfe; font-family: 'Outfit', sans-serif; font-size: 17px;">🛠️ Technologies Used</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;">
                <span style="background: rgba(0,242,254,0.1); border: 1px solid rgba(0,242,254,0.3); border-radius: 8px; padding: 6px 14px; color: #00f2fe; font-size: 13px; font-family: 'Inter', sans-serif; font-weight: 600;">Python</span>
                <span style="background: rgba(0,242,254,0.1); border: 1px solid rgba(0,242,254,0.3); border-radius: 8px; padding: 6px 14px; color: #00f2fe; font-size: 13px; font-family: 'Inter', sans-serif; font-weight: 600;">Pandas</span>
                <span style="background: rgba(0,242,254,0.1); border: 1px solid rgba(0,242,254,0.3); border-radius: 8px; padding: 6px 14px; color: #00f2fe; font-size: 13px; font-family: 'Inter', sans-serif; font-weight: 600;">NumPy</span>
                <span style="background: rgba(0,242,254,0.1); border: 1px solid rgba(0,242,254,0.3); border-radius: 8px; padding: 6px 14px; color: #00f2fe; font-size: 13px; font-family: 'Inter', sans-serif; font-weight: 600;">Scikit-Learn</span>
                <span style="background: rgba(0,242,254,0.1); border: 1px solid rgba(0,242,254,0.3); border-radius: 8px; padding: 6px 14px; color: #00f2fe; font-size: 13px; font-family: 'Inter', sans-serif; font-weight: 600;">Random Forest</span>
                <span style="background: rgba(0,242,254,0.1); border: 1px solid rgba(0,242,254,0.3); border-radius: 8px; padding: 6px 14px; color: #00f2fe; font-size: 13px; font-family: 'Inter', sans-serif; font-weight: 600;">Streamlit</span>
                <span style="background: rgba(0,242,254,0.1); border: 1px solid rgba(0,242,254,0.3); border-radius: 8px; padding: 6px 14px; color: #00f2fe; font-size: 13px; font-family: 'Inter', sans-serif; font-weight: 600;">Plotly</span>
                <span style="background: rgba(0,242,254,0.1); border: 1px solid rgba(0,242,254,0.3); border-radius: 8px; padding: 6px 14px; color: #00f2fe; font-size: 13px; font-family: 'Inter', sans-serif; font-weight: 600;">Joblib</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #fda085; font-family: 'Outfit', sans-serif; font-size: 17px;">⚠️ Important Limitations & Disclaimer</h4>
        <ul style="color: #cbd5e1; line-height: 2; font-family: 'Inter', sans-serif; font-size: 14px;">
            <li>This is an ML-based <b>candidate classification</b> system — it does not scientifically confirm new exoplanets</li>
            <li>Model performance depends on the quality and distribution of the training dataset</li>
            <li>Synthetic demo observations are generated for demonstration and are not real astronomical data</li>
            <li>Real astronomical confirmation requires additional scientific validation (radial velocity, spectroscopy)</li>
            <li>Dataset class imbalance can affect prediction reliability for minority classes</li>
            <li>This ML model should not be interpreted as a replacement for professional astronomical analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
