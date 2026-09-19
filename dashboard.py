"""
Spotter – Freight Rate Intelligence Dashboard
==============================================
Minimalist, light-mode-first Streamlit app for freight rate prediction
and logistics analytics. Plain language, clean design.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    st.set_page_config(page_title="Spotter | Setup Required", page_icon="🚚")
    st.error(
        "**Setup required:** Run with `uv run streamlit run dashboard.py`"
    )
    st.stop()

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotter | Rate Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths & Constants ──────────────────────────────────────────────────────────
DATA_DIR = Path("data")
TRAIN_FILE = DATA_DIR / "train-test.csv"
VALID_FILE = DATA_DIR / "validation.csv"
DECEMBER_FILE = DATA_DIR / "december-chart-inputs.csv"
SUBMISSION_FILE = Path("validation_predictions.csv")
SCORER_IMAGE = Path("scorer_results/candidate_december.png")

CAT_FEATURES = ["pickup", "delivery", "equipment"]
DECEMBER_PICKUP = "Lexington"
DECEMBER_DELIVERY = "Fort Wayne"
DECEMBER_PICKUP_LAT, DECEMBER_PICKUP_LON = 36.99152, -84.99876
DECEMBER_DELIVERY_LAT, DECEMBER_DELIVERY_LON = 41.31561, -85.36206

EQUIPMENT_LABELS = {
    "Dry Van": "Dry Van",
    "Flatbed": "Flatbed",
    "Reefer": "Refrigerated",
}

# ── CSS: theme tokens, visible controls, spacing ──────────────────────────────
def _theme_tokens(dark: bool) -> str:
    if dark:
        return """
        --bg: #0b1220;
        --bg-elevated: #121a2b;
        --surface: #172033;
        --surface-hover: #1d2942;
        --text: #f3f4f6;
        --text-muted: #d1d5db;
        --text-faint: #94a3b8;
        --border: #2a3750;
        --border-strong: #3b4d6b;
        --accent: #60a5fa;
        --accent-hover: #93c5fd;
        --accent-soft: rgba(96,165,250,0.22);
        --success: #4ade80;
        --success-hover: #86efac;
        --btn-fg: #0b1220;
        --pill-bg: #e2e8f0;
        --pill-fg: #0f172a;
        --stepper-bg: #334155;
        --shadow: 0 8px 24px rgba(0,0,0,0.35);
        --header-grad: linear-gradient(145deg, #1e3a8a 0%, #0f172a 100%);
        --plot-bg: #121a2b;
        --plot-grid: #243044;
        --plot-font: #e5e7eb;
        --chip-bg: #1e3a5f;
        --chip-fg: #dbeafe;
        --tab-hover-bg: #93c5fd;
        --tab-hover-fg: #0b1220;
        --tab-fg: #ffffff;
        --tab-bg: #334155;
        --focus: 0 0 0 3px rgba(147,197,253,0.45);
        """
    return """
        --bg: #ffffff;
        --bg-elevated: #f8fafc;
        --surface: #ffffff;
        --surface-hover: #f8fafc;
        --text: #111827;
        --text-muted: #4b5563;
        --text-faint: #9ca3af;
        --border: #e2e8f0;
        --border-strong: #cbd5e1;
        --accent: #2563eb;
        --accent-hover: #1d4ed8;
        --accent-soft: rgba(37,99,235,0.12);
        --success: #16a34a;
        --success-hover: #15803d;
        --btn-fg: #ffffff;
        --pill-bg: #ffffff;
        --pill-fg: #111827;
        --stepper-bg: #ffffff;
        --shadow: 0 1px 4px rgba(15,23,42,0.06);
        --header-grad: linear-gradient(145deg, #111827 0%, #1e3a5f 100%);
        --plot-bg: #f8fafc;
        --plot-grid: #eef2f7;
        --plot-font: #1a1a2e;
        --chip-bg: #eef2ff;
        --chip-fg: #3730a3;
        --tab-hover-bg: #dbeafe;
        --tab-hover-fg: #1e3a8a;
        --tab-fg: #1f2937;
        --tab-bg: #ffffff;
        --focus: 0 0 0 3px rgba(37,99,235,0.18);
        """


def inject_css(dark: bool) -> None:
    tokens = _theme_tokens(dark)
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root, [data-theme="light"], [data-theme="dark"] {{
    {tokens}
}}

html, body, [class*="css"], .stApp, .stMarkdown, p, span, label {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.stApp {{
    background: var(--bg) !important;
    color: var(--text) !important;
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
.stDeployButton, [data-testid="stToolbarActionButton"] {{ display: none; }}
header[data-testid="stHeader"],
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stToolbar"] {{
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}}

.block-container {{
    padding-top: 1.6rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1.75rem !important;
    padding-right: 1.75rem !important;
    max-width: 1180px !important;
}}

[data-testid="stHorizontalBlock"] {{
    gap: 1rem !important;
    align-items: stretch;
}}

/* Tabs — do not inherit action-button styles */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: transparent;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
}}
.stTabs [data-baseweb="tab"],
.stTabs button,
.stTabs [role="tab"] {{
    background: var(--tab-bg) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 18px !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: var(--tab-fg) !important;
    opacity: 1 !important;
    min-height: 2.5rem !important;
    box-shadow: none !important;
}}
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
.stTabs button p,
.stTabs button span,
.stTabs [role="tab"] p,
.stTabs [role="tab"] span {{
    color: var(--tab-fg) !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}}
.stTabs [data-baseweb="tab"]:hover,
.stTabs button:hover,
.stTabs [role="tab"]:hover {{
    background: var(--tab-hover-bg) !important;
    color: var(--tab-hover-fg) !important;
    opacity: 1 !important;
}}
.stTabs [data-baseweb="tab"]:hover p,
.stTabs [data-baseweb="tab"]:hover span,
.stTabs button:hover p,
.stTabs button:hover span,
.stTabs [role="tab"]:hover p,
.stTabs [role="tab"]:hover span {{
    color: var(--tab-hover-fg) !important;
    opacity: 1 !important;
}}
.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span {{
    color: var(--btn-fg) !important;
    font-weight: 800 !important;
    opacity: 1 !important;
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}}
.stTabs [aria-selected="true"]:hover,
.stTabs [aria-selected="true"]:hover p,
.stTabs [aria-selected="true"]:hover span {{
    background: var(--tab-hover-bg) !important;
    color: var(--tab-hover-fg) !important;
}}

.card, .card-sm {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow);
}}
.card {{ padding: 20px 22px; margin-bottom: 14px; }}
.card-sm {{ padding: 16px 18px; height: 100%; }}

.kpi-label {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-faint);
    margin-bottom: 6px;
}}
.kpi-value {{
    font-size: 1.85rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
    line-height: 1.15;
}}
.kpi-value-accent {{ color: var(--accent); }}
.kpi-value-green  {{ color: var(--success); }}
.kpi-sub {{ font-size: 0.78rem; color: var(--text-faint); margin-top: 4px; }}

.section-header {{
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text);
    margin: 4px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}}

.result-box {{
    background: var(--header-grad);
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: var(--shadow);
    border: 1px solid rgba(255,255,255,0.08);
}}
.result-label {{
    font-size: 0.78rem;
    font-weight: 600;
    color: rgba(255,255,255,0.65);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 8px;
}}
.result-value {{
    font-size: 2.75rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1;
}}
.result-sub {{ font-size: 0.9rem; color: rgba(255,255,255,0.5); margin-top: 10px; }}
.result-range {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 11px 16px;
    margin-top: 16px;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.78);
}}

.summary-table td.muted {{ color: var(--text-muted); }}
.summary-table td.val {{ color: var(--text); font-weight: 600; text-align: right; }}

/* Action buttons only — never style tabs / steppers / toolbar */
[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button {{
    background: var(--accent) !important;
    color: var(--btn-fg) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.45rem 1.15rem !important;
    min-height: 2.5rem !important;
    line-height: 1.3 !important;
    opacity: 1 !important;
    box-shadow: none !important;
}}
[data-testid="stButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {{
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
    color: var(--btn-fg) !important;
}}
[data-testid="stButton"] button:focus-visible,
[data-testid="stDownloadButton"] button:focus-visible {{
    outline: none !important;
    box-shadow: var(--focus) !important;
}}
[data-testid="stButton"] button p,
[data-testid="stButton"] button span {{
    color: var(--btn-fg) !important;
    font-weight: 700 !important;
}}
[data-testid="stDownloadButton"] button p,
[data-testid="stDownloadButton"] button span {{
    color: var(--btn-fg) !important;
    font-weight: 700 !important;
}}

[data-testid="stDownloadButton"] button {{
    background: var(--success) !important;
    color: var(--btn-fg) !important;
    border: 1px solid var(--success) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    min-height: 2.5rem !important;
    padding: 0.45rem 1.15rem !important;
    opacity: 1 !important;
}}
[data-testid="stButton"] button:hover p,
[data-testid="stButton"] button:hover span,
[data-testid="stFormSubmitButton"] button:hover p,
[data-testid="stFormSubmitButton"] button:hover span {{
    color: var(--btn-fg) !important;
}}
[data-testid="stDownloadButton"] button:hover {{
    background: var(--success-hover) !important;
    border-color: var(--success-hover) !important;
    color: var(--btn-fg) !important;
}}
[data-testid="stDownloadButton"] button:hover p,
[data-testid="stDownloadButton"] button:hover span {{
    color: var(--btn-fg) !important;
}}

/* Number / date stepper buttons */
[data-testid="stNumberInput"] button,
[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"],
[data-testid="stDateInput"] button {{
    background: var(--stepper-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    opacity: 1 !important;
    min-width: 1.85rem !important;
    min-height: 1.85rem !important;
}}
[data-testid="stNumberInput"] button svg,
[data-testid="stNumberInputStepUp"] svg,
[data-testid="stNumberInputStepDown"] svg,
[data-testid="stDateInput"] button svg {{
    fill: var(--text) !important;
    stroke: var(--text) !important;
}}

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
.stSelectbox label, .stNumberInput label, .stDateInput label,
.stSlider label, .stMultiSelect label, .stTextInput label, .stRadio label {{
    color: var(--text-muted) !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
}}

[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextInput"] input {{
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
}}
[data-baseweb="select"]:focus-within > div,
[data-testid="stNumberInput"] input:focus,
[data-testid="stDateInput"] input:focus {{
    border-color: var(--accent) !important;
    box-shadow: var(--focus) !important;
}}

[data-testid="stSlider"] span {{
    color: var(--text-muted) !important;
}}
[data-testid="stSlider"] [role="slider"] {{
    background: var(--accent) !important;
}}

div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {{
    color: var(--text) !important;
    font-size: 0.88rem !important;
}}

span[data-baseweb="tag"] {{
    background: var(--chip-bg) !important;
    border-radius: 6px !important;
}}
span[data-baseweb="tag"] > span {{ color: var(--chip-fg) !important; font-weight: 600 !important; }}
span[data-baseweb="tag"] button svg {{ fill: var(--chip-fg) !important; }}

div[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: var(--shadow);
}}
div[data-testid="stMetricLabel"] p {{
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}
div[data-testid="stMetricValue"] {{ color: var(--text) !important; }}
div[data-testid="stMetricValue"] p {{
    color: var(--text) !important;
    font-weight: 700 !important;
    font-size: 1.4rem !important;
}}

div[data-testid="stExpander"] {{
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--surface) !important;
}}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] p {{
    color: var(--text) !important;
}}
div[data-testid="stExpander"] summary:hover,
div[data-testid="stExpander"] summary:hover p,
div[data-testid="stExpander"] summary:hover span {{
    color: var(--tab-hover-fg) !important;
    background: var(--tab-hover-bg) !important;
}}

[data-testid="stSidebar"] {{
    background: var(--surface);
    border-right: 1px solid var(--border);
}}

[data-testid="stDataFrame"] {{
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}}

.app-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 4px 0 16px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}}
.app-logo {{
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    flex-shrink: 0;
}}
.app-title {{
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
    line-height: 1.2;
}}
.app-subtitle {{ font-size: 0.82rem; color: var(--text-muted); margin-top: 2px; }}

[data-testid="stToggle"] label p {{
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
}}

[data-testid="stPills"] button,
[data-testid="stSegmentedControl"] button {{
    background: var(--pill-bg) !important;
    color: var(--pill-fg) !important;
    border: 1.5px solid var(--border-strong) !important;
    min-height: 2.25rem !important;
    opacity: 1 !important;
    box-shadow: none !important;
    font-weight: 700 !important;
}}
[data-testid="stPills"] button p,
[data-testid="stSegmentedControl"] button p,
[data-testid="stPills"] button span,
[data-testid="stSegmentedControl"] button span {{
    color: var(--pill-fg) !important;
}}
[data-testid="stPills"] button[aria-checked="true"],
[data-testid="stPills"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[aria-checked="true"] {{
    background: var(--accent) !important;
    color: var(--btn-fg) !important;
    border-color: var(--accent) !important;
}}
[data-testid="stPills"] button[aria-checked="true"] p,
[data-testid="stPills"] button[aria-pressed="true"] p,
[data-testid="stPills"] button[aria-checked="true"] span,
[data-testid="stPills"] button[aria-pressed="true"] span,
[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
[data-testid="stSegmentedControl"] button[aria-checked="true"] span {{
    color: var(--btn-fg) !important;
}}
[data-testid="stPills"] button:hover,
[data-testid="stSegmentedControl"] button:hover {{
    background: var(--accent-hover) !important;
    color: var(--btn-fg) !important;
    border-color: var(--accent-hover) !important;
}}
[data-testid="stPills"] button:hover p,
[data-testid="stPills"] button:hover span,
[data-testid="stSegmentedControl"] button:hover p,
[data-testid="stSegmentedControl"] button:hover span {{
    color: var(--btn-fg) !important;
}}

@media (max-width: 768px) {{
    .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}
    .result-value {{ font-size: 2.1rem; }}
    .kpi-value {{ font-size: 1.45rem; }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, dark: bool) -> go.Figure:
    font_color = "#e5e7eb" if dark else "#1a1a2e"
    plot_bg = "#121a2b" if dark else "#f8fafc"
    grid = "#243044" if dark else "#eef2f7"
    line = "#2a3750" if dark else "#e2e8f0"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bg,
        font=dict(family="Inter, sans-serif", size=12, color=font_color),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=font_color)),
        margin=dict(l=8, r=8, t=12, b=8),
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid, linecolor=line, zeroline=False, color=font_color)
    fig.update_yaxes(showgrid=True, gridcolor=grid, linecolor=line, zeroline=False, color=font_color)
    return fig


# ── Feature Engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, city_coords: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    if city_coords is not None:
        if "pickup_lat" not in df.columns or df["pickup_lat"].isna().all():
            df["pickup_lat"] = df["pickup"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[0])
            df["pickup_lon"] = df["pickup"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[1])
        if "delivery_lat" not in df.columns or df["delivery_lat"].isna().all():
            df["delivery_lat"] = df["delivery"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[0])
            df["delivery_lon"] = df["delivery"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[1])

    lat1 = np.radians(df["pickup_lat"])
    lon1 = np.radians(df["pickup_lon"])
    lat2 = np.radians(df["delivery_lat"])
    lon2 = np.radians(df["delivery_lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["haversine_miles"] = 3956.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    df["bearing"] = np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )

    dist = np.maximum(0, df["distance"])
    df["log_distance"] = np.log1p(dist)
    df["sqrt_distance"] = np.sqrt(dist)
    df["dist_diff"] = df["distance"] - df["haversine_miles"]
    df["circuitous_ratio"] = df["distance"] / (df["haversine_miles"] + 1.0)

    w = np.maximum(0, df["weight"].fillna(df["weight"].median()))
    df["log_weight"] = np.log1p(w)
    df["dist_weight"] = df["distance"] * w
    df["weight_per_mile"] = w / (df["distance"] + 1e-5)

    if "market_index" in df.columns and "quote_signal" in df.columns:
        df["market_x_quote"] = df["market_index"] * df["quote_signal"]
        df["market_x_dist"] = df["market_index"] * df["distance"]
        df["quote_x_dist"] = df["quote_signal"] * df["distance"]
    else:
        df["market_index"] = np.nan
        df["quote_signal"] = np.nan
        df["market_x_quote"] = np.nan
        df["market_x_dist"] = np.nan
        df["quote_x_dist"] = np.nan

    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    for col in CAT_FEATURES:
        df[col] = df[col].astype("category")

    return df


def add_target_encodings(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame | None,
    target_col: str,
    group_cols: list[str],
    prior_weight: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    global_mean = train_df[target_col].mean()
    encodings: dict = {}
    for col in group_cols:
        stats = train_df.groupby(col, observed=False)[target_col].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + prior_weight * global_mean) / (
            stats["count"] + prior_weight
        )
        enc_dict = smooth.to_dict()
        encodings[col] = (enc_dict, global_mean)
        feature_name = f"{col}_te_rpm"
        train_df[feature_name] = train_df[col].map(enc_dict).fillna(global_mean)
        if infer_df is not None:
            infer_df[feature_name] = infer_df[col].map(enc_dict).fillna(global_mean)
    return train_df, infer_df, encodings


def get_features(df: pd.DataFrame) -> list[str]:
    exclude = {"load_id", "date", "posted_rate", "rate_per_mile", "lane"}
    return [c for c in df.columns if c not in exclude]


# ── Cached Data & Model ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not TRAIN_FILE.exists() or not VALID_FILE.exists():
        st.error(f"Data files not found in `{DATA_DIR}/`.")
        st.stop()

    train_df = pd.read_csv(TRAIN_FILE)
    valid_df = pd.read_csv(VALID_FILE)
    dec_df = pd.read_csv(DECEMBER_FILE) if DECEMBER_FILE.exists() else None

    train_df["date"] = pd.to_datetime(train_df["date"])
    train_df["month"] = train_df["date"].dt.month
    train_df["rate_per_mile"] = train_df["posted_rate"] / train_df["distance"]
    train_df["lane"] = train_df["pickup"] + " → " + train_df["delivery"]

    valid_df["date"] = pd.to_datetime(valid_df["date"])
    valid_df["month"] = valid_df["date"].dt.month
    valid_df["lane"] = valid_df["pickup"] + " → " + valid_df["delivery"]

    city_coords: dict = {}
    for df_, lat_col, lon_col, city_col in [
        (train_df, "pickup_lat", "pickup_lon", "pickup"),
        (train_df, "delivery_lat", "delivery_lon", "delivery"),
        (valid_df, "pickup_lat", "pickup_lon", "pickup"),
        (valid_df, "delivery_lat", "delivery_lon", "delivery"),
    ]:
        if lat_col in df_.columns:
            sub = df_.dropna(subset=[lat_col, lon_col]).drop_duplicates(subset=[city_col])
            for _, row in sub.iterrows():
                city_coords[row[city_col]] = (row[lat_col], row[lon_col])

    city_coords.setdefault(DECEMBER_PICKUP, (DECEMBER_PICKUP_LAT, DECEMBER_PICKUP_LON))
    city_coords.setdefault(DECEMBER_DELIVERY, (DECEMBER_DELIVERY_LAT, DECEMBER_DELIVERY_LON))

    train_eng = engineer_features(train_df, city_coords)
    valid_eng = engineer_features(valid_df, city_coords)

    if SUBMISSION_FILE.exists():
        preds = pd.read_csv(SUBMISSION_FILE)
        pred_map = dict(zip(preds["load_id"], preds["predicted_rate"]))
        valid_df["predicted_rate"] = valid_df["load_id"].map(pred_map)
    else:
        valid_df["predicted_rate"] = np.nan

    return train_df, valid_df, dec_df, train_eng, valid_eng, city_coords


@st.cache_resource
def train_model():
    train_df, valid_df, _, train_eng, valid_eng, city_coords = load_data()

    full_train = train_eng.copy()
    valid_copy = valid_eng.copy()

    full_train, valid_copy, encodings = add_target_encodings(
        full_train, valid_copy, "rate_per_mile",
        group_cols=["pickup", "delivery", "equipment"]
    )

    feats = get_features(full_train)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=700,
        learning_rate=0.035,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.5,
        categorical_features=CAT_FEATURES,
        random_state=42,
        n_iter_no_change=50,
        validation_fraction=0.05,
    )
    model.fit(full_train[feats], full_train["rate_per_mile"])
    return model, feats, encodings, city_coords


# Load everything
train_df, valid_df, dec_df, train_eng, valid_eng, city_coords = load_data()
model, feats, te_encodings, city_coords = train_model()

# ── App Header + theme ─────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

DARK = bool(st.session_state.dark_mode)
inject_css(DARK)

hdr_left, hdr_right = st.columns([7, 2], vertical_alignment="center")
with hdr_left:
    st.markdown(
        """
        <div class="app-header" style="border-bottom:none; margin-bottom:0; padding-bottom:0;">
            <div class="app-logo">🚚</div>
            <div>
                <div class="app-title">Spotter</div>
                <div class="app-subtitle">Freight Rate Intelligence · North America</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hdr_right:
    st.toggle(
        "Dark mode",
        key="dark_mode",
        help="Switch between light and dark color schemes",
    )

# ── Main Tabs ──────────────────────────────────────────────────────────────────
tab_predict, tab_market, tab_december, tab_data = st.tabs(
    ["🎯 Rate Predictor", "📊 Market Overview", "🗓️ December Forecast", "📁 Data"]
)


# ==============================================================================
# TAB 1 — RATE PREDICTOR (default / first tab)
# ==============================================================================
with tab_predict:
    col_form, col_result = st.columns([1.15, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">Shipment Details</div>', unsafe_allow_html=True)

        all_cities = sorted(list(city_coords.keys()))

        if st.session_state.pop("_swap_cities", False):
            pickup_val = st.session_state.get("pickup")
            delivery_val = st.session_state.get("delivery")
            if pickup_val and delivery_val:
                st.session_state.pickup, st.session_state.delivery = delivery_val, pickup_val

        r1, r2 = st.columns(2)
        with r1:
            pickup_city = st.selectbox(
                "Pickup City",
                options=all_cities,
                index=all_cities.index("Lexington") if "Lexington" in all_cities else 0,
                key="pickup",
            )
        with r2:
            delivery_city = st.selectbox(
                "Delivery City",
                options=all_cities,
                index=all_cities.index("Fort Wayne") if "Fort Wayne" in all_cities else min(1, len(all_cities) - 1),
                key="delivery",
            )

        if st.button("Swap pickup & delivery", help="Reverse the lane", key="swap_cities"):
            st.session_state._swap_cities = True
            st.rerun()

        try:
            equip = st.pills(
                "Trailer Type",
                options=["Dry Van", "Flatbed", "Reefer"],
                default="Dry Van",
                key="equip",
            )
        except Exception:
            equip = st.selectbox(
                "Trailer Type",
                options=["Dry Van", "Flatbed", "Reefer"],
                format_func=lambda x: EQUIPMENT_LABELS.get(x, x),
                key="equip",
            )
        if equip is None:
            equip = "Dry Van"

        # Auto-calculate distance from coordinates
        p_lat, p_lon = city_coords.get(pickup_city, (36.99, -84.99))
        d_lat, d_lon = city_coords.get(delivery_city, (41.31, -85.36))
        lat1, lon1, lat2, lon2 = map(np.radians, [p_lat, p_lon, d_lat, d_lon])
        dlat_r, dlon_r = lat2 - lat1, lon2 - lon1
        a_r = np.sin(dlat_r / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon_r / 2) ** 2
        calc_miles = float(3956.0 * 2 * np.arcsin(np.sqrt(np.clip(a_r, 0, 1))))
        default_miles = max(50.0, round(calc_miles * 1.15, 0))

        r3, r4 = st.columns(2)
        with r3:
            dist_val = st.number_input(
                "Distance (miles)",
                min_value=10.0,
                max_value=3000.0,
                value=default_miles,
                step=10.0,
                key="dist",
            )
        with r4:
            weight_val = st.number_input(
                "Weight (lbs)",
                min_value=100.0,
                max_value=50000.0,
                value=32000.0,
                step=500.0,
                key="weight",
            )

        load_date = st.date_input(
            "Pickup Date",
            value=pd.to_datetime("2025-12-15"),
            key="load_date",
        )

        with st.expander("Market Conditions (optional)", expanded=False):
            mkt_idx = st.slider(
                "Demand Level",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
                help="Current freight market demand. 1.0 = average, >1 = high demand",
                key="mkt",
            )
            quote_sig = st.slider(
                "Spot Quote Pressure",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
                help="How competitive the current spot market is",
                key="quote",
            )

    with col_result:
        # ── Run inference ──
        input_data = pd.DataFrame([{
            "load_id": "CALC-001",
            "pickup": pickup_city,
            "delivery": delivery_city,
            "equipment": equip,
            "distance": dist_val,
            "weight": weight_val,
            "date": str(load_date),
            "market_index": mkt_idx,
            "quote_signal": quote_sig,
            "pickup_lat": p_lat,
            "pickup_lon": p_lon,
            "delivery_lat": d_lat,
            "delivery_lon": d_lon,
        }])

        input_eng = engineer_features(input_data, city_coords)

        for col in ["pickup", "delivery", "equipment"]:
            enc_dict, global_mean = te_encodings.get(col, ({}, 2.15))
            val = input_eng[col].iloc[0]
            input_eng[f"{col}_te_rpm"] = enc_dict.get(val, global_mean)

        pred_rpm = float(model.predict(input_eng[feats])[0])
        pred_rate = pred_rpm * dist_val
        lower = max(0, pred_rate - 115.0)
        upper = pred_rate + 115.0
        rate_per_mile = pred_rpm

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Estimated Freight Rate</div>
                <div class="result-value">${pred_rate:,.0f}</div>
                <div class="result-sub">{pickup_city} &nbsp;→&nbsp; {delivery_city}</div>
                <div class="result-range">
                    Expected range: <strong>${lower:,.0f} – ${upper:,.0f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        # Secondary metrics
        m1, m2 = st.columns(2, gap="medium")
        with m1:
            st.markdown(
                f"""
                <div class="card-sm">
                    <div class="kpi-label">Per Mile</div>
                    <div class="kpi-value kpi-value-accent">${rate_per_mile:.2f}</div>
                    <div class="kpi-sub">per mile</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="card-sm">
                    <div class="kpi-label">Trip Distance</div>
                    <div class="kpi-value">{dist_val:,.0f}</div>
                    <div class="kpi-sub">miles</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Route summary
        st.markdown(
            f"""
            <div class="card" style="padding: 16px 20px;">
                <div class="section-header" style="font-size:0.9rem; margin-bottom:10px;">Trip Summary</div>
                <div style="display:flex; justify-content:space-between; padding:7px 0; font-size:0.88rem; border-bottom:1px solid var(--border);">
                    <span class="kpi-sub" style="margin:0;">Route</span>
                    <span style="font-weight:600; color:var(--text);">{pickup_city} → {delivery_city}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:7px 0; font-size:0.88rem; border-bottom:1px solid var(--border);">
                    <span class="kpi-sub" style="margin:0;">Trailer</span>
                    <span style="font-weight:600; color:var(--text);">{EQUIPMENT_LABELS.get(equip, equip)}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:7px 0; font-size:0.88rem; border-bottom:1px solid var(--border);">
                    <span class="kpi-sub" style="margin:0;">Weight</span>
                    <span style="font-weight:600; color:var(--text);">{weight_val:,.0f} lbs</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:7px 0; font-size:0.88rem;">
                    <span class="kpi-sub" style="margin:0;">Pickup Date</span>
                    <span style="font-weight:600; color:var(--text);">{str(load_date)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 2 — MARKET OVERVIEW
# ==============================================================================
with tab_market:
    # KPI row
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    avg_rate = train_df["posted_rate"].mean()
    median_rpm = train_df["rate_per_mile"].median()
    total_loads = len(train_df)
    unique_lanes = train_df["lane"].nunique()

    with k1:
        st.markdown(
            f"""
            <div class="card-sm">
                <div class="kpi-label">Avg Freight Rate</div>
                <div class="kpi-value">${avg_rate:,.0f}</div>
                <div class="kpi-sub">across all loads</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="card-sm">
                <div class="kpi-label">Median Rate / Mile</div>
                <div class="kpi-value kpi-value-accent">${median_rpm:.2f}</div>
                <div class="kpi-sub">per mile</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="card-sm">
                <div class="kpi-label">Total Loads</div>
                <div class="kpi-value">{total_loads:,}</div>
                <div class="kpi-sub">historical shipments</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="card-sm">
                <div class="kpi-label">Unique Routes</div>
                <div class="kpi-value kpi-value-green">{unique_lanes:,}</div>
                <div class="kpi-sub">city pairs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Filters (inline, minimal)
    fc1, fc2 = st.columns([1.15, 1.35], gap="medium")
    with fc1:
        equip_filter = st.multiselect(
            "Trailer Type",
            options=["Dry Van", "Flatbed", "Reefer"],
            default=["Dry Van", "Flatbed", "Reefer"],
            key="mkt_equip",
        )
    with fc2:
        dist_range = st.slider(
            "Distance (miles)",
            min_value=int(train_df["distance"].min()),
            max_value=int(train_df["distance"].max()),
            value=(int(train_df["distance"].min()), int(train_df["distance"].max())),
            step=50,
            key="mkt_dist",
        )

    train_filtered = train_df[
        train_df["equipment"].isin(equip_filter)
        & (train_df["distance"] >= dist_range[0])
        & (train_df["distance"] <= dist_range[1])
    ].copy()

    st.markdown("")

    if train_filtered.empty:
        st.info("No loads match these filters. Adjust trailer type or distance.")
    else:
        ch1, ch2 = st.columns(2, gap="large")
        with ch1:
            st.markdown('<div class="section-header">Rate by Distance</div>', unsafe_allow_html=True)
            sample = train_filtered.sample(min(3000, len(train_filtered)), random_state=42)
            fig1 = px.scatter(
                sample,
                x="distance",
                y="posted_rate",
                color="equipment",
                hover_data=["pickup", "delivery"],
                opacity=0.55,
                labels={"distance": "Distance (miles)", "posted_rate": "Freight Rate ($)"},
                color_discrete_map={"Dry Van": "#2563eb", "Flatbed": "#f59e0b", "Reefer": "#10b981"},
                template="simple_white",
            )
            fig1.update_layout(height=340)
            st.plotly_chart(style_fig(fig1, DARK), width="stretch")

        with ch2:
            st.markdown('<div class="section-header">Rate Distribution by Trailer</div>', unsafe_allow_html=True)
            fig2 = px.histogram(
                train_filtered,
                x="rate_per_mile",
                color="equipment",
                nbins=50,
                barmode="overlay",
                opacity=0.75,
                labels={"rate_per_mile": "Rate per Mile ($/mile)"},
                color_discrete_map={"Dry Van": "#2563eb", "Flatbed": "#f59e0b", "Reefer": "#10b981"},
                template="simple_white",
            )
            fig2.update_layout(height=340)
            st.plotly_chart(style_fig(fig2, DARK), width="stretch")

        st.markdown('<div class="section-header">Monthly Rate Trend</div>', unsafe_allow_html=True)
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly = (
            train_filtered.groupby(["month", "equipment"], observed=False)["posted_rate"]
            .mean()
            .reset_index()
        )
        monthly["Month"] = monthly["month"].map(month_names)
        fig3 = px.line(
            monthly,
            x="Month",
            y="posted_rate",
            color="equipment",
            markers=True,
            labels={"posted_rate": "Avg Rate ($)"},
            color_discrete_map={"Dry Van": "#2563eb", "Flatbed": "#f59e0b", "Reefer": "#10b981"},
            template="simple_white",
        )
        fig3.update_layout(height=300)
        st.plotly_chart(style_fig(fig3, DARK), width="stretch")


# ==============================================================================
# TAB 3 — DECEMBER FORECAST
# ==============================================================================
with tab_december:
    st.markdown(
        """
        <div class="card" style="padding:16px 22px; margin-bottom:16px;">
            <div class="kpi-label">Fixed Scenario</div>
            <div class="kpi-value" style="font-size:1.15rem;">
                Lexington → Fort Wayne · Dry Van · 360 mi · 32,000 lbs
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if dec_df is not None:
        dec_df["date"] = pd.to_datetime(dec_df["date"])
        dec_df["rate_per_mile"] = dec_df["predicted_rate"] / dec_df["distance"]

        d1, d2, d3, d4 = st.columns(4, gap="medium")
        with d1:
            st.metric("Lowest Rate", f"${dec_df['predicted_rate'].min():,.2f}")
        with d2:
            st.metric("Highest Rate", f"${dec_df['predicted_rate'].max():,.2f}")
        with d3:
            st.metric("Average Rate", f"${dec_df['predicted_rate'].mean():,.2f}")
        with d4:
            st.metric("Day-to-Day Swing", f"±${dec_df['predicted_rate'].std():,.2f}")

        st.markdown("")

        fig_dec = px.line(
            dec_df,
            x="date",
            y="predicted_rate",
            markers=True,
            labels={"date": "Date", "predicted_rate": "Predicted Rate ($)"},
            color_discrete_sequence=["#3b82f6" if DARK else "#2563eb"],
            template="simple_white",
        )
        fig_dec.update_traces(line=dict(width=2.5), marker=dict(size=7))
        fig_dec.update_layout(height=380)
        st.plotly_chart(style_fig(fig_dec, DARK), width="stretch")

        if SCORER_IMAGE.exists():
            st.markdown('<div class="section-header">Official Scorer Output</div>', unsafe_allow_html=True)
            st.image(str(SCORER_IMAGE), width=800)
    else:
        st.info("December forecast data not found. Expected at `data/december-chart-inputs.csv`.")


# ==============================================================================
# TAB 4 — DATA
# ==============================================================================
with tab_data:
    try:
        view = st.segmented_control(
            "Dataset",
            options=["Predictions", "Training History"],
            default="Predictions",
            key="data_view",
        )
    except Exception:
        view = st.radio(
            "Dataset",
            options=["Predictions", "Training History"],
            horizontal=True,
            key="data_view",
        )

    if view == "Predictions":
        st.markdown('<div class="section-header">Validation Predictions</div>', unsafe_allow_html=True)
        display_cols = ["load_id", "pickup", "delivery", "equipment", "distance", "weight", "date", "predicted_rate"]
        cols_present = [c for c in display_cols if c in valid_df.columns]
        st.dataframe(valid_df[cols_present].head(500), width="stretch", hide_index=True)

        if SUBMISSION_FILE.exists():
            st.download_button(
                label="Download Predictions CSV",
                data=SUBMISSION_FILE.read_bytes(),
                file_name="validation_predictions.csv",
                mime="text/csv",
                type="primary",
            )
    else:
        st.markdown('<div class="section-header">Training Data (Jan–Oct 2025)</div>', unsafe_allow_html=True)
        show_cols = ["load_id", "pickup", "delivery", "equipment", "distance", "weight", "posted_rate", "date"]
        cols_present = [c for c in show_cols if c in train_df.columns]
        st.dataframe(train_df[cols_present].head(500), width="stretch", hide_index=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="kpi-sub" style="text-align:center; padding: 28px 0 8px 0;">
        Spotter · Freight Rate Intelligence · 2025
    </div>
    """,
    unsafe_allow_html=True,
)
