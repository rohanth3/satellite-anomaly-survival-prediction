"""
Satellite Anomaly Time-to-Event (TTE) Predictor
Interactive Web Application & Intelligence Dashboard
Authors: Rohanth Ganesha C (PES1UG24AM227) & Poorvi Rangaswamy (PES1UG24AM193)
Department of Computer Science and Engineering, PES University
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure ModifiedNaiveBayesModel is discoverable for pickle loading
from run_pipeline import ModifiedNaiveBayesModel
from src.hybrid_ensemble import HybridSurvivalEnsemble
sys.modules['__main__'].ModifiedNaiveBayesModel = ModifiedNaiveBayesModel

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Satellite Anomaly TTE Predictor | PES University",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING (Modern Space-Themed Dashboard)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header hero styling */
    .hero-banner {
        background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.18) 0%, rgba(14, 165, 233, 0.12) 50%, rgba(15, 23, 42, 0) 100%),
                    linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px rgba(99, 102, 241, 0.15);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.02rem;
        margin-bottom: 16px;
    }
    .author-pill {
        display: inline-block;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        color: #e2e8f0;
        margin-right: 8px;
        margin-top: 4px;
    }
    .author-pill strong {
        color: #38bdf8;
    }

    /* Model Result Cards */
    .model-card {
        background: linear-gradient(145deg, #131b2e 0%, #0f172a 100%);
        border-radius: 14px;
        padding: 20px 22px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .model-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.45);
    }
    .model-card.featured {
        border: 1.5px solid rgba(16, 185, 129, 0.5);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.15);
    }
    .model-card.hybrid {
        border: 1.5px solid rgba(245, 158, 11, 0.5);
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.15);
    }
    .model-tag {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .tag-baseline {
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    .tag-extension {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .tag-hybrid {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .model-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 6px;
    }
    .model-tte {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 12px 0 6px 0;
    }
    .tte-unit {
        font-size: 0.95rem;
        font-weight: 500;
        color: #94a3b8;
    }
    .model-meta {
        font-size: 0.8rem;
        color: #94a3b8;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 10px;
        margin-top: 10px;
    }

    /* Risk status banner */
    .status-alert {
        padding: 16px 22px;
        border-radius: 12px;
        margin: 20px 0;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .status-alert.critical {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #fca5a5;
    }
    .status-alert.elevated {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fde68a;
    }
    .status-alert.nominal {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #a7f3d0;
    }
    
    /* Telemetry spec card */
    .telemetry-badge {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .telemetry-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .telemetry-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RESOURCE & DATA CACHING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pre-trained machine learning checkpoints...")
def load_models():
    """Load serialized models, scalers, and ensemble estimators."""
    scaler = joblib.load("models/scaler.joblib")
    lr = joblib.load("models/linear_regression.joblib")
    svr = joblib.load("models/svr.joblib")
    nb = joblib.load("models/modified_naive_bayes.joblib")
    gbr = joblib.load("models/gradient_boosting.joblib")
    ens = joblib.load("models/hybrid_ensemble.joblib")
    return {
        "scaler": scaler,
        "lr": lr,
        "svr": svr,
        "nb": nb,
        "gbr": gbr,
        "ens": ens
    }

@st.cache_data
def load_datasets():
    """Load dataset, orbital parameters, and benchmark tables."""
    data_df = pd.read_csv("data.csv")
    satfeat_df = pd.read_csv("satfeat.csv")
    metrics_df = pd.read_csv("results/metrics_summary.csv")
    pred_df = pd.read_csv("results/predictions_comparison.csv")
    return data_df, satfeat_df, metrics_df, pred_df

models_dict = load_models()
data_df, satfeat_df, metrics_df, pred_df = load_datasets()

# -----------------------------------------------------------------------------
# SATELLITE PRESETS CATALOG
# -----------------------------------------------------------------------------
PRESET_SATELLITES = {
    "Select a Satellite Preset or Customize Below...": {
        "perigee": 20200.0,
        "inclination": 55.0,
        "ssn": 80.0,
        "month": 6,
        "desc": "Custom Telemetry (Defaults: Mid-altitude MEO satellite with moderate solar activity)"
    },
    "🛰️ Meteosat 1 (GEO Weather Satellite)": {
        "perigee": 35725.0,
        "inclination": 4.82,
        "ssn": 52.0,
        "month": 5,
        "desc": "Geostationary meteorological satellite in Earth-synchronous equatorial orbit."
    },
    "🛰️ GOES 4 (GEO Weather Spacecraft)": {
        "perigee": 35951.0,
        "inclination": 7.77,
        "ssn": 78.0,
        "month": 4,
        "desc": "NOAA Geostationary Operational Environmental Satellite for severe weather monitoring."
    },
    "🛰️ GOES 5 (GEO Weather Spacecraft)": {
        "perigee": 35780.0,
        "inclination": 9.18,
        "ssn": 64.0,
        "month": 8,
        "desc": "High-altitude geostationary craft constantly exposed to solar particle events."
    },
    "🛰️ Marecs A (GEO Maritime Communications)": {
        "perigee": 36795.0,
        "inclination": 11.51,
        "ssn": 95.0,
        "month": 9,
        "desc": "High perigee geostationary telecommunications satellite for trans-oceanic vessel links."
    },
    "🛰️ NOAA 11 (LEO Polar Earth Observation)": {
        "perigee": 831.0,
        "inclination": 98.63,
        "ssn": 142.0,
        "month": 11,
        "desc": "Low Earth Orbit sun-synchronous craft traversing auroral radiation belts and polar cusps."
    },
    "🛰️ NAVSTAR 3 / GPS 5113 (MEO Constellation)": {
        "perigee": 20545.0,
        "inclination": 62.86,
        "ssn": 88.0,
        "month": 6,
        "desc": "Global Positioning System satellite in Van Allen radiation inner-outer boundary zone."
    },
    "🛰️ NAVSTAR 4 / GPS 5114 (MEO Constellation)": {
        "perigee": 20483.0,
        "inclination": 63.78,
        "ssn": 115.0,
        "month": 3,
        "desc": "Block II GPS navigation payload with high event frequency in the NOAA catalog."
    },
    "🛰️ NAVSTAR 6 / GPS 5118 (MEO Constellation)": {
        "perigee": 18470.0,
        "inclination": 61.81,
        "ssn": 42.0,
        "month": 1,
        "desc": "Medium altitude GPS bird operating in deep radiation belt conditions."
    },
    "🛰️ NAVSTAR 8 / GPS 9794 (MEO Constellation)": {
        "perigee": 20378.0,
        "inclination": 63.51,
        "ssn": 65.0,
        "month": 7,
        "desc": "Core NAVSTAR spacecraft with regular operational single-event anomalies."
    },
    "⚡ Extreme Solar Max Event (Space Weather Storm)": {
        "perigee": 35786.0,
        "inclination": 0.1,
        "ssn": 240.0,
        "month": 10,
        "desc": "Severe solar storm simulation: high sunspot count with heavy geomagnetic radiation injection."
    },
    "🌍 Low Earth Orbit (ISS / Starlink Regime)": {
        "perigee": 550.0,
        "inclination": 53.2,
        "ssn": 110.0,
        "month": 2,
        "desc": "Atmospheric drag & South Atlantic Anomaly (SAA) high-flux environment."
    }
}

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Telemetry Input Panel")
st.sidebar.markdown("Choose a verified spacecraft profile or enter custom orbital parameters.")

preset_choice = st.sidebar.selectbox("Satellite Profile Preset:", list(PRESET_SATELLITES.keys()))
default_vals = PRESET_SATELLITES[preset_choice]

st.sidebar.caption(default_vals["desc"])
st.sidebar.markdown("---")

# Feature Inputs
perigee = st.sidebar.number_input(
    "Perigee Altitude (km):",
    min_value=300.0,
    max_value=45000.0,
    value=float(default_vals["perigee"]),
    step=100.0,
    help="Lowest orbital altitude above Earth's surface (km). Determines atmospheric and radiation belt exposure."
)

# Orbit regime indicator
if perigee < 2000:
    orbit_badge = "LEO (Low Earth Orbit)"
elif perigee < 35000:
    orbit_badge = "MEO (Medium Earth Orbit)"
else:
    orbit_badge = "GEO (Geostationary Orbit)"

st.sidebar.caption(f"**Orbital Regime:** `{orbit_badge}`")

inclination = st.sidebar.slider(
    "Orbital Inclination (degrees):",
    min_value=0.0,
    max_value=120.0,
    value=float(default_vals["inclination"]),
    step=0.1,
    help="Angle of orbit relative to Earth's equator. High inclinations (>60°) traverse energetic polar horns."
)

ssn = st.sidebar.slider(
    "Daily Sunspot Number (SSN):",
    min_value=0.0,
    max_value=300.0,
    value=float(default_vals["ssn"]),
    step=1.0,
    help="International Sunspot Number on interval start date. Higher values correspond to intense solar storms."
)

if ssn < 50:
    solar_activity = "Quiet Solar Conditions"
elif ssn < 130:
    solar_activity = "Moderate Solar Activity"
else:
    solar_activity = "High Activity / Solar Storm"

st.sidebar.caption(f"**Solar State:** `{solar_activity}`")

month = st.sidebar.slider(
    "Month of Year (1 - 12):",
    min_value=1,
    max_value=12,
    value=int(default_vals["month"]),
    step=1,
    help="Month corresponding to the start of the observation interval (1 = Jan, 12 = Dec)."
)

month_names = ["January", "February", "March", "April", "May", "June", 
               "July", "August", "September", "October", "November", "December"]
st.sidebar.caption(f"**Calendar Period:** `{month_names[month - 1]}`")

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🚀 Calculate Anomaly TTE", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# TOP HERO BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛰️ Satellite Anomaly Time-to-Event (TTE) Predictor</div>
    <div class="hero-subtitle">
        Operational Mission Telemetry Intelligence: Estimating Days Until the Next Spacecraft Anomaly 
        via Survival Analysis and Gradient-Boosted Decision Ensembles.
    </div>
    <div>
        <span class="author-pill">🎓 <strong>PES University</strong> | AI & ML</span>
        <span class="author-pill">👨‍💻 <strong>Rohanth Ganesha C</strong> (PES1UG24AM227)</span>
        <span class="author-pill">👩‍💻 <strong>Poorvi Rangaswamy</strong> (PES1UG24AM193)</span>
        <span class="author-pill">🔗 <a href="https://github.com/rohanth3/satellite-anomaly-survival-prediction" target="_blank" style="color: #38bdf8; text-decoration: none;">GitHub Repository</a></span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PREDICTION ENGINE EXECUTION
# -----------------------------------------------------------------------------
raw_x = np.array([[perigee, inclination, ssn, month]])
scaler = models_dict["scaler"]
scaled_x = scaler.transform(raw_x)

# Model 1: Linear Regression
pred_lr = float(models_dict["lr"].predict(scaled_x)[0])
pred_lr_clipped = max(1.0, pred_lr)

# Model 2: Support Vector Regression
pred_svr = float(models_dict["svr"].predict(scaled_x)[0])
pred_svr_clipped = max(1.0, pred_svr)

# Model 3: Modified Naive Bayes with KM Prior
pred_nb = float(models_dict["nb"].predict(scaled_x)[0])

# Model 4: Gradient Boosting (L1 Loss)
pred_gbr = float(np.clip(models_dict["gbr"].predict(raw_x)[0], 1.0, 365.0))

# Model 5: Hybrid Survival Ensemble
pred_ens = float(models_dict["ens"].predict(raw_x)[0])

# Survival Probability from Kaplan-Meier Prior
km_prior_dict = models_dict["nb"].km_prior
sorted_times = sorted(km_prior_dict.keys())
# Find closest survival prob
closest_time = min(sorted_times, key=lambda t: abs(t - pred_ens))
survival_prob = km_prior_dict[closest_time] * 100.0

# -----------------------------------------------------------------------------
# NAVIGATION TABS
# -----------------------------------------------------------------------------
tab_pred, tab_arch, tab_bench, tab_eda, tab_data = st.tabs([
    "🚀 Live Anomaly Predictor",
    "🏛️ Architecture & Formulation",
    "📊 Benchmark Comparisons",
    "📈 Research Figures & EDA",
    "🔍 Dataset & Spacecraft Catalog"
])

# =============================================================================
# TAB 1: LIVE ANOMALY PREDICTOR
# =============================================================================
with tab_pred:
    st.markdown("### 📡 Active Satellite Telemetry Summary")
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown(f"""
        <div class="telemetry-badge">
            <div class="telemetry-val">{perigee:,.0f} km</div>
            <div class="telemetry-lbl">Perigee ({orbit_badge.split(' ')[0]})</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        st.markdown(f"""
        <div class="telemetry-badge">
            <div class="telemetry-val">{inclination:.1f}°</div>
            <div class="telemetry-lbl">Inclination</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t3:
        st.markdown(f"""
        <div class="telemetry-badge">
            <div class="telemetry-val">{ssn:.0f} SSN</div>
            <div class="telemetry-lbl">Solar Activity</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t4:
        st.markdown(f"""
        <div class="telemetry-badge">
            <div class="telemetry-val">{month_names[month - 1][:3]} (M{month})</div>
            <div class="telemetry-lbl">Observation Period</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Multi-Model Time-to-Event Predictions")
    st.caption("Estimated days remaining until the next recorded physical anomaly across all baseline and proposed models:")

    # 5 Model Result Columns
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        st.markdown(f"""
        <div class="model-card">
            <div>
                <span class="model-tag tag-baseline">Baseline 1</span>
                <div class="model-name">Linear Regression</div>
                <div class="model-tte" style="color: #94a3b8;">{pred_lr_clipped:.1f} <span class="tte-unit">days</span></div>
            </div>
            <div class="model-meta">
                Test MAE: <b>21.50 d</b><br>
                Test MRE: <b>493.5%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="model-card">
            <div>
                <span class="model-tag tag-baseline">Baseline 2</span>
                <div class="model-name">Support Vector (SVR)</div>
                <div class="model-tte" style="color: #f43f5e;">{pred_svr_clipped:.1f} <span class="tte-unit">days</span></div>
            </div>
            <div class="model-meta">
                Test MAE: <b>17.22 d</b><br>
                Kernel: <b>RBF (C=10.0)</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="model-card">
            <div>
                <span class="model-tag tag-baseline">Baseline 3</span>
                <div class="model-name">Modified Naive Bayes</div>
                <div class="model-tte" style="color: #a855f7;">{pred_nb:.1f} <span class="tte-unit">days</span></div>
            </div>
            <div class="model-meta">
                Test MAE: <b>17.19 d</b><br>
                Prior: <b>Kaplan-Meier</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="model-card featured">
            <div>
                <span class="model-tag tag-extension">Proposed Extension 🏆</span>
                <div class="model-name">Gradient Boosting (L1)</div>
                <div class="model-tte" style="color: #34d399;">{pred_gbr:.1f} <span class="tte-unit">days</span></div>
            </div>
            <div class="model-meta">
                Test MAE: <b style="color: #34d399;">16.73 d (Lowest)</b><br>
                Loss: <b>Absolute Error</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col5:
        st.markdown(f"""
        <div class="model-card hybrid">
            <div>
                <span class="model-tag tag-hybrid">Proposed Extension 🌟</span>
                <div class="model-name">Hybrid Survival Ensemble</div>
                <div class="model-tte" style="color: #fbbf24;">{pred_ens:.1f} <span class="tte-unit">days</span></div>
            </div>
            <div class="model-meta">
                Test MRE: <b style="color: #fbbf24;">165.0% (Best)</b><br>
                Blend: <b>70% GBR + 30% KM</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Operational Alert Level Banner based on Hybrid Prediction
    if pred_ens <= 7.0:
        st.markdown(f"""
        <div class="status-alert critical">
            <span style="font-size: 1.8rem;">🚨</span>
            <div>
                <strong style="font-size: 1.05rem;">CRITICAL ANOMALY ALERT — HIGH RISK HORIZON ({pred_ens:.1f} DAYS)</strong><br>
                The predicted time to next anomaly is under 1 week. Immediate ground station telemetry monitoring, payload thermal safeguards, and safe-mode contingency readiness recommended.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif pred_ens <= 21.0:
        st.markdown(f"""
        <div class="status-alert elevated">
            <span style="font-size: 1.8rem;">⚠️</span>
            <div>
                <strong style="font-size: 1.05rem;">ELEVATED RISK HORIZON — INTERMEDIATE WINDOW ({pred_ens:.1f} DAYS)</strong><br>
                The satellite is operating within an elevated risk window (1 to 3 weeks). Schedule non-essential payload updates and monitor space weather flux notifications.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-alert nominal">
            <span style="font-size: 1.8rem;">✅</span>
            <div>
                <strong style="font-size: 1.05rem;">NOMINAL OPERATIONAL MARGIN ({pred_ens:.1f} DAYS)</strong><br>
                Estimated failure interval exceeds 3 weeks. Routine telemetry tracking and standard housekeeping commands in progress.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Visual Comparison Bar Chart & Kaplan-Meier Gauge
    col_plot1, col_plot2 = st.columns([3, 2])

    with col_plot1:
        st.markdown("#### 📊 Prediction Concordance Comparison")
        model_names_plot = ["Linear Reg", "SVR (RBF)", "KM Bayes", "GBR (L1)", "Hybrid Ensemble"]
        model_preds_plot = [pred_lr_clipped, pred_svr_clipped, pred_nb, pred_gbr, pred_ens]
        colors_plot = ["#64748b", "#f43f5e", "#a855f7", "#10b981", "#f59e0b"]

        fig, ax = plt.subplots(figsize=(8, 4.2))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')

        bars = ax.bar(model_names_plot, model_preds_plot, color=colors_plot, width=0.55, edgecolor="#475569")
        ax.axhline(np.mean(model_preds_plot), color="#38bdf8", linestyle="--", linewidth=1.2, label=f"Mean: {np.mean(model_preds_plot):.1f} d")
        ax.axhline(np.median(model_preds_plot), color="#fbbf24", linestyle=":", linewidth=1.2, label=f"Median: {np.median(model_preds_plot):.1f} d")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f} d",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', color="#ffffff", fontweight='bold', fontsize=10)

        ax.set_ylabel("Predicted Days to Next Event", color="#94a3b8", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.2, color='#94a3b8')
        ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#e2e8f0', loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_plot2:
        st.markdown("#### ⏱️ Kaplan-Meier Survival Probability")
        st.markdown(f"""
        Based on the empirical non-parametric Kaplan-Meier product-limit estimator derived from **890 historical satellite anomaly intervals**:
        """)

        st.metric(
            label="Estimated Survival Probability S(t)",
            value=f"{survival_prob:.1f}%",
            delta=f"at t = {pred_ens:.1f} days",
            help="Probability that a satellite survives at least t days without experiencing an anomaly."
        )

        st.progress(min(1.0, max(0.0, survival_prob / 100.0)))
        st.caption(f"There is a **{survival_prob:.1f}% chance** that the satellite will operate beyond **{pred_ens:.1f} days** before this anomaly event takes place.")

        st.markdown("##### 💡 Evaluator Takeaway:")
        st.info(
            "Notice how **Gradient Boosting (L1)** and **Support Vector Regression** cluster near the empirical median (5–7 days). "
            "Linear Regression overestimates the interval because it is heavily dragged upward by the rare extreme outliers (e.g. 346-day intervals)."
        )

# =============================================================================
# TAB 2: ARCHITECTURE & METHODOLOGY
# =============================================================================
with tab_arch:
    st.markdown("### 🏛️ System Architecture & Problem Formulation")

    col_a1, col_a2 = st.columns([1, 1])

    with col_a1:
        st.markdown("#### 1. Time-to-Event (TTE) Formulation")
        st.write("""
        Conventional anomaly detection treats anomalies as binary flags:
        $$\\text{State} \\in \\{0, 1\\} \\quad \\text{(\"Will an anomaly occur?\")}$$
        
        This binary formulation provides no operational timeline. In contrast, this project formulates anomaly prediction as a **continuous Time-to-Event (TTE) regression problem**:
        """)
        st.latex(r"\text{TTE}_i = t_{i+1} - t_i")
        st.write("""
        where $t_i$ and $t_{i+1}$ are consecutive recorded anomaly dates for a given satellite. 
        Each interval is treated as an independent observation paired with satellite orbital and space-weather features:
        """)
        st.latex(r"X = [\text{Perigee (km)}, \text{Inclination (deg)}, \text{Sunspot Number (SSN)}, \text{Month}]")

    with col_a2:
        st.markdown("#### 2. Multi-Source Data Ingestion & Preprocessing")
        st.write("""
        Raw satellite telemetry was harmonized across three disparate scientific providers:
        - **NOAA Satellite Anomaly Database (`anom5j.xls`)**: Historical anomaly incident dates and telemetry (`BIRD`, `ADATE`).
        - **CelesTrak SATCAT (`satcat.csv`)**: Satellite catalog numbers and physical orbital parameters (Perigee, Inclination).
        - **SILSO Sunspot Index (`sunspot.csv`)**: Daily international sunspot counts recording solar magnetic activity.
        
        **GPS Satellite Name Mapping Problem**:
        NOAA catalogs GPS birds by their internal tracking codes (`gps 5113`), whereas CelesTrak catalogs them under operational military designations (`NAVSTAR 3 (OPS 5113)`). 
        The pipeline normalizes these identifiers to ensure zero missing orbital mappings across all GPS spacecraft.
        """)

    st.markdown("---")
    st.markdown("#### 3. Mathematical Foundations of the 5 Evaluated Models")

    m_row1_col1, m_row1_col2 = st.columns(2)

    with m_row1_col1:
        st.markdown("##### 1. Linear Regression (Baseline)")
        st.write("""
        Fits an ordinary least squares hyperplane:
        $$\\hat{y} = w^T X + b$$
        Provides the baseline test for linear correlation between environmental features and failure intervals. Highly vulnerable to right-skewed target distributions.
        """)

        st.markdown("##### 2. Support Vector Regression (SVR Baseline)")
        st.write("""
        Uses an $\\epsilon$-insensitive loss function and Radial Basis Function (RBF) kernel:
        $$K(x, x') = \\exp(-\\gamma ||x - x'||^2)$$
        Allows nonlinear boundary fitting. With $C=10.0$ and $\\epsilon=0.1$, SVR ignores small errors within the margin tube.
        """)

    with m_row1_col2:
        st.markdown("##### 3. Modified Naive Bayes with Kaplan-Meier Prior (Baseline)")
        st.write("""
        Combines Gaussian feature likelihoods with a non-parametric Kaplan-Meier survival prior:
        $$S(t) = \\prod_{t_i \\leq t} \\left(1 - \\frac{d_i}{n_i}\\right)$$
        $$P(t \\mid X) \\propto S(t) \\cdot \\prod_{j=1}^4 P(X_j \\mid \\text{Event at } t)$$
        Predicts the time $t$ that maximizes the posterior likelihood.
        """)

        st.markdown("##### 4. Gradient Boosting Regressor (Proposed Extension)")
        st.write("""
        Trains an additive sequence of shallow decision trees minimizing **L1 absolute error loss**:
        $$\\mathcal{L}(y, \\hat{y}) = |y - \\hat{y}|$$
        L1 loss fits conditional medians rather than means, shielding the model from extreme long-interval outliers and achieving our best MAE (16.73 days).
        """)

    st.markdown("##### 5. Hybrid Survival-Boosting Ensemble (Proposed Extension)")
    st.write("""
    Combines discriminative tree-based boosting with empirical statistical survival distributions via a convex combination:
    $$\\hat{y}_{\\text{hybrid}} = 0.70 \\cdot \\hat{y}_{\\text{GBR}} + 0.30 \\cdot \\hat{y}_{\\text{NB-KM}}$$
    This model achieves the lowest Mean Relative Error (**165.0%**) while maintaining an exceptional MAE (**16.77 days**).
    """)

# =============================================================================
# TAB 3: BENCHMARK COMPARISONS
# =============================================================================
with tab_bench:
    st.markdown("### 📊 10-Fold Cross-Validation Performance Benchmarks")
    st.write("All models were rigorously evaluated using 10-fold shuffled cross-validation (`random_state=42`) across the complete 890-event dataset:")

    # Highlight metrics
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Lowest Mean Absolute Error (MAE)", "16.73 days", "Gradient Boosting (L1)")
    h2.metric("Lowest Median Error (MedAE)", "5.24 days", "Gradient Boosting (L1)")
    h3.metric("Lowest Relative Error (MRE)", "165.0%", "Hybrid Survival Ensemble")
    h4.metric("Lowest Root Mean Squared Error", "39.05 days", "Linear Regression")

    # Table
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Comprehensive Benchmark Visualizations")

    col_g1, col_g2 = st.columns([1, 1])
    with col_g1:
        if os.path.exists("results/model_comparison.png"):
            st.image("results/model_comparison.png", caption="Figure 1: Benchmark Error Comparison (MAE, RMSE, and MRE across all 5 models)")
    with col_g2:
        if os.path.exists("results/predictions_scatter.png"):
            st.image("results/predictions_scatter.png", caption="Figure 2: Predicted TTE vs True TTE with y = x Reference Fit (Replication of Paper Fig 2)")

    st.markdown("---")
    st.markdown("### 🔍 Understanding the \"Relative Error Paradox\" (Critical for Viva & Evaluation)")
    
    st.markdown("""
    > **Why is the Mean Relative Error (MRE) ~165% while the Mean Absolute Error is only ~16.7 days and Median Error is only ~5.2 days?**
    """)

    st.write("""
    The apparent discrepancy is caused by the **small denominator effect** in the formula for Mean Relative Error:
    """)
    st.latex(r"\text{MRE} = \frac{1}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%")
    st.write("""
    1. **Target Distribution Skew**: In our dataset of 890 events, **124 anomalies occurred on Day 1** ($y=1$) and **88 occurred on Day 2** ($y=2$).
    2. **Extreme Percentage Penalty on Small Denominators**:
       - If an anomaly occurs on Day 1 ($y_i = 1$) and the model predicts 5 days ($\hat{y}_i = 5$), the absolute error is only **4 days**, but the relative error is:
    """)
    st.latex(r"\left| \frac{1 - 5}{1} \right| \times 100\% = 400\%")
    st.write("""
       - If an anomaly occurs on Day 2 ($y_i = 2$) and the model predicts 8 days ($\hat{y}_i = 8$), the relative error is **300%**.
    3. **Conclusion for Spacecraft Operators**: 
       A high relative error percentage does **not** signify failure; in day-to-day operations, predicting that an anomaly will happen in 5 days instead of 1 day is a practical difference of only **4 days**. The Median Absolute Error of **5.2 days** confirms that more than half of all predictions are within ~5 days of ground truth!
    """)

# =============================================================================
# TAB 4: RESEARCH VISUALIZATIONS & EDA
# =============================================================================
with tab_eda:
    st.markdown("### 📈 Scientific Exploratory Data Analysis & Empirical Distributions")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        if os.path.exists("results/kaplan_meier_survival_curve.png"):
            st.image("results/kaplan_meier_survival_curve.png", caption="Figure 3: Non-Parametric Kaplan-Meier Survival Probability Curve S(t)")
            st.caption("""
            **Interpretation:** The empirical probability of remaining anomaly-free decays rapidly within the first 30 days. By Day 30, survival probability drops below 20%, showing that satellite anomalies cluster strongly in time.
            """)

    with eda_col2:
        if os.path.exists("results/tte_distribution.png"):
            st.image("results/tte_distribution.png", caption="Figure 4: Distribution of Time-to-Event Target Variable (Days)")
            st.caption("""
            **Interpretation:** Severe right-skewed exponential-like distribution. Most failure intervals are short (1–15 days), while occasional calm periods extend out to 300+ days.
            """)

    st.markdown("<br>", unsafe_allow_html=True)
    eda_col3, eda_col4 = st.columns(2)

    with eda_col3:
        if os.path.exists("results/feature_importance.png"):
            st.image("results/feature_importance.png", caption="Figure 5: Gradient Boosting Gini Feature Importance")
            st.caption("""
            **Interpretation:** Daily Sunspot Number (`ssn`, ~48%) and Perigee Altitude (`perigee`, ~34%) dominate prediction importance. Solar activity triggers geomagnetic storms, while perigee dictates the radiation belt intensity.
            """)

    with eda_col4:
        if os.path.exists("results/feature_correlations.png"):
            st.image("results/feature_correlations.png", caption="Figure 6: Pearson Feature Correlation Heatmap")
            st.caption("""
            **Interpretation:** Shows low collinearity between orbital altitude (`perigee`), orbital inclination (`inclination`), solar flux (`ssn`), and seasonal months (`month`).
            """)

# =============================================================================
# TAB 5: DATASET & SPACECRAFT CATALOG
# =============================================================================
with tab_data:
    st.markdown("### 🛰️ Spacecraft Telemetry & Training Dataset Explorer")

    d_stat1, d_stat2, d_stat3, d_stat4 = st.columns(4)
    d_stat1.metric("Total Valid Events", f"{len(data_df):,} intervals")
    d_stat2.metric("Unique Satellites", f"{data_df['BIRD'].nunique()} spacecraft")
    d_stat3.metric("Orbital Features", f"{len(satfeat_df.columns) - 1} physical parameters")
    d_stat4.metric("Filtered TTE Range", f"{data_df['TTE'].min():.0f} – {data_df['TTE'].max():.0f} days")

    st.markdown("#### 1. Spacecraft Orbital Features Catalog (`satfeat.csv`)")
    st.dataframe(satfeat_df, use_container_width=True)

    st.markdown("#### 2. Event Records Table (`data.csv`)")
    selected_bird = st.selectbox("Filter by Spacecraft (`BIRD`):", ["All Satellites"] + sorted(data_df["BIRD"].unique()))
    
    if selected_bird == "All Satellites":
        filtered_df = data_df
    else:
        filtered_df = data_df[data_df["BIRD"] == selected_bird]

    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("#### 3. Summary Statistics of Final Merged Dataset")
    merged_preview = data_df.merge(satfeat_df.rename(columns={"Name": "BIRD"}), on="BIRD", how="inner")
    st.dataframe(merged_preview[["TTE", "Perigee (km)", "Inclination (deg)", "SSN", "Month"]].describe(), use_container_width=True)

# -----------------------------------------------------------------------------
# FOOTER BANNER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0 24px 0;">
    <b>Satellite Anomaly Survival Prediction System</b> — Machine Learning & Survival Analysis<br>
    Developed by <b>Rohanth Ganesha C</b> (PES1UG24AM227) & <b>Poorvi Rangaswamy</b> (PES1UG24AM193)<br>
    Department of Computer Science & Engineering, PES University, Bangalore.<br>
    Source Code: <a href="https://github.com/rohanth3/satellite-anomaly-survival-prediction" target="_blank" style="color: #38bdf8;">github.com/rohanth3/satellite-anomaly-survival-prediction</a>
</div>
""", unsafe_allow_html=True)
