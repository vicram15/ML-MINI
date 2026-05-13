import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataML Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --accent: #00FFB2;
    --accent2: #FF6B6B;
    --accent3: #6C63FF;
    --bg-dark: #0A0E1A;
    --bg-card: #111827;
    --bg-card2: #1a2236;
    --text-primary: #F0F4FF;
    --text-muted: #7C8DB5;
    --border: #1E2D4A;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text-primary);
}

.main { background: var(--bg-dark); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0A0E1A 0%, #111827 50%, #0d1b2e 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,255,178,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -1px;
    margin: 0 0 0.3rem 0;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-muted);
    margin: 0;
    font-weight: 300;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    flex: 1; min-width: 140px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent3));
    border-radius: 0 0 12px 12px;
}
.metric-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.2px; font-family: 'Space Mono', monospace; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: var(--accent); margin-top: 0.2rem; }

/* Section headers */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.5rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Info boxes */
.info-box {
    background: rgba(0,255,178,0.06);
    border: 1px solid rgba(0,255,178,0.2);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
}
.warn-box {
    background: rgba(255,107,107,0.06);
    border: 1px solid rgba(255,107,107,0.2);
    border-left: 3px solid var(--accent2);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00d4a1) !important;
    color: #0A0E1A !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,255,178,0.3) !important;
}
.stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio label { color: var(--text-muted) !important; font-size: 0.82rem !important; }
div[data-testid="stSelectbox"] > div { background: var(--bg-card2) !important; border-color: var(--border) !important; }
.stDataFrame { background: var(--bg-card2); border-radius: 10px; }
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card2); border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: var(--text-muted) !important; font-family: 'Space Mono', monospace !important; font-size: 0.78rem !important; }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: #0A0E1A !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="#0A0E1A",
    plot_bgcolor="#111827",
    font=dict(family="DM Sans", color="#F0F4FF"),
    colorway=["#00FFB2", "#6C63FF", "#FF6B6B", "#FFD166", "#06D6A0", "#EF476F"],
    xaxis=dict(gridcolor="#1E2D4A", linecolor="#1E2D4A"),
    yaxis=dict(gridcolor="#1E2D4A", linecolor="#1E2D4A"),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_TEMPLATE)
    return fig

def card(label, value, unit=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}<span style="font-size:0.9rem;color:#7C8DB5"> {unit}</span></div>
    </div>"""

def section(title, icon=""):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)

def infer_task(series):
    if series.dtype == object or series.nunique() <= 10:
        return "classification"
    return "regression"

def encode_df(df):
    df = df.copy()
    le_map = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_map[col] = le
    return df, le_map

CLASSIFIERS = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
}

REGRESSORS = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "SVR": SVR(),
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-family:'Space Mono',monospace; font-size:1.4rem; color:#00FFB2; font-weight:700;">⚡ DataML</div>
        <div style="font-size:0.72rem; color:#7C8DB5; letter-spacing:2px; text-transform:uppercase;">Studio</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Upload your dataset as a .csv file")

    if uploaded:
        sep = st.selectbox("Delimiter", [",", ";", "\t", "|"], index=0)
        st.markdown("---")

    st.markdown("""
    <div style="font-size:0.72rem; color:#7C8DB5; margin-top:2rem; line-height:1.7;">
    <b style="color:#00FFB2">FEATURES</b><br>
    📊 Auto EDA<br>
    🧹 Data Cleaning<br>
    🤖 Auto ML<br>
    📈 Feature Importance<br>
    🔮 Live Predictions
    </div>
    """, unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">⚡ DataML Studio</div>
    <div class="hero-sub">Upload any CSV · Explore · Train ML models · Predict — all in one place</div>
</div>
""", unsafe_allow_html=True)

# ── No file ────────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("""
    <div class="info-box">
    👈  <strong>Upload a CSV file</strong> from the sidebar to get started. The app will automatically
    analyse your data, detect the ML task type, train multiple models and let you make live predictions.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card" style="text-align:center;padding:2rem 1rem;">
            <div style="font-size:2rem">📊</div>
            <div style="font-family:'Space Mono',monospace;color:#00FFB2;margin-top:.5rem;font-size:.85rem">AUTO EDA</div>
            <div style="font-size:.8rem;color:#7C8DB5;margin-top:.4rem">Distributions, correlations & missing value analysis</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card" style="text-align:center;padding:2rem 1rem;">
            <div style="font-size:2rem">🤖</div>
            <div style="font-family:'Space Mono',monospace;color:#00FFB2;margin-top:.5rem;font-size:.85rem">AUTO ML</div>
            <div style="font-size:.8rem;color:#7C8DB5;margin-top:.4rem">Trains & compares 5 models simultaneously</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card" style="text-align:center;padding:2rem 1rem;">
            <div style="font-size:2rem">🔮</div>
            <div style="font-family:'Space Mono',monospace;color:#00FFB2;margin-top:.5rem;font-size:.85rem">PREDICT</div>
            <div style="font-size:.8rem;color:#7C8DB5;margin-top:.4rem">Live predictions using the best trained model</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file, sep):
    return pd.read_csv(file, sep=sep)

df = load_data(uploaded, sep)

# ── Dataset overview ───────────────────────────────────────────────────────────
rows, cols = df.shape
missing = df.isnull().sum().sum()
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()
dup = df.duplicated().sum()

st.markdown('<div class="metric-row">' +
    card("Rows", f"{rows:,}") +
    card("Columns", cols) +
    card("Numeric", len(num_cols)) +
    card("Categorical", len(cat_cols)) +
    card("Missing", f"{missing:,}") +
    card("Duplicates", dup) +
'</div>', unsafe_allow_html=True)

tabs = st.tabs(["🔍 Explore", "📊 Visualize", "🤖 ML Training", "🔮 Predict"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section("Dataset Preview", "📋")
    st.dataframe(df.head(50), use_container_width=True, height=280)

    c1, c2 = st.columns(2)
    with c1:
        section("Data Types & Missing Values", "🔎")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.count().values,
            "Missing": df.isnull().sum().values,
            "Missing %": (df.isnull().mean() * 100).round(2).values,
            "Unique": df.nunique().values,
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with c2:
        section("Statistical Summary", "📐")
        st.dataframe(df.describe().T.round(3), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VISUALIZE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    v1, v2 = st.columns(2)

    with v1:
        section("Distribution", "📈")
        if num_cols:
            sel_dist = st.selectbox("Select column", num_cols, key="dist")
            fig = px.histogram(df, x=sel_dist, nbins=40, color_discrete_sequence=["#00FFB2"],
                               marginal="violin", template="plotly_dark")
            apply_theme(fig)
            fig.update_layout(height=320, margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

    with v2:
        section("Correlation Heatmap", "🔥")
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig2 = px.imshow(corr, color_continuous_scale=["#0A0E1A", "#6C63FF", "#00FFB2"],
                             text_auto=".2f", template="plotly_dark")
            apply_theme(fig2)
            fig2.update_layout(height=320, margin=dict(t=20,b=20))
            st.plotly_chart(fig2, use_container_width=True)

    v3, v4 = st.columns(2)
    with v3:
        section("Scatter Plot", "✦")
        if len(num_cols) >= 2:
            sx = st.selectbox("X axis", num_cols, key="sx")
            sy = st.selectbox("Y axis", [c for c in num_cols if c != sx], key="sy")
            color_col = st.selectbox("Color by", ["None"] + cat_cols + num_cols, key="sc")
            fig3 = px.scatter(df, x=sx, y=sy,
                              color=None if color_col == "None" else color_col,
                              template="plotly_dark", opacity=0.7,
                              color_discrete_sequence=["#00FFB2","#6C63FF","#FF6B6B"])
            apply_theme(fig3)
            fig3.update_layout(height=320, margin=dict(t=20,b=20))
            st.plotly_chart(fig3, use_container_width=True)

    with v4:
        section("Box Plot", "📦")
        if num_cols:
            bx = st.selectbox("Numeric column", num_cols, key="bx")
            by = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="by")
            fig4 = px.box(df, x=None if by == "None" else by, y=bx,
                          template="plotly_dark", color=None if by == "None" else by,
                          color_discrete_sequence=["#00FFB2","#6C63FF","#FF6B6B","#FFD166"])
            apply_theme(fig4)
            fig4.update_layout(height=320, margin=dict(t=20,b=20))
            st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section("Model Configuration", "⚙️")

    ml1, ml2, ml3 = st.columns(3)
    with ml1:
        target = st.selectbox("🎯 Target Column", df.columns.tolist())
    with ml2:
        task = infer_task(df[target])
        task_override = st.radio("Task Type", ["Auto-detect", "Classification", "Regression"], horizontal=True)
        if task_override != "Auto-detect":
            task = task_override.lower()
    with ml3:
        test_size = st.slider("Test Size %", 10, 40, 20) / 100

    # Feature selection
    all_features = [c for c in df.columns if c != target]
    features = st.multiselect("📌 Feature Columns (default = all)", all_features, default=all_features)

    if not features:
        st.markdown('<div class="warn-box">⚠️ Please select at least one feature column.</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown(f'<div class="info-box">🤖 Task detected as <strong>{task.upper()}</strong> — '
                f'training on <strong>{len(features)}</strong> features, '
                f'target: <strong>{target}</strong></div>', unsafe_allow_html=True)

    run_btn = st.button("🚀 Train All Models")

    if run_btn:
        with st.spinner("Training models…"):
            # Prep
            subset = df[features + [target]].dropna(subset=[target])
            X = subset[features]
            y = subset[target]

            # Encode
            X_enc, _ = encode_df(X)
            if task == "classification":
                le_target = LabelEncoder()
                y_enc = le_target.fit_transform(y.astype(str))
            else:
                y_enc = pd.to_numeric(y, errors="coerce")
                mask = ~np.isnan(y_enc)
                X_enc, y_enc = X_enc[mask], y_enc[mask]

            # Impute
            imp = SimpleImputer(strategy="median")
            X_enc = pd.DataFrame(imp.fit_transform(X_enc), columns=X_enc.columns)

            # Scale
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X_enc)

            X_train, X_test, y_train, y_test = train_test_split(
                X_sc, y_enc, test_size=test_size, random_state=42,
                stratify=y_enc if task == "classification" else None)

            models = CLASSIFIERS if task == "classification" else REGRESSORS
            results = {}
            trained = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if task == "classification":
                    results[name] = {
                        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                        "Predictions": y_pred,
                    }
                else:
                    results[name] = {
                        "R²": round(r2_score(y_test, y_pred), 4),
                        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
                        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                        "Predictions": y_pred,
                    }
                trained[name] = model

            st.session_state["results"] = results
            st.session_state["trained"] = trained
            st.session_state["task"] = task
            st.session_state["features"] = features
            st.session_state["target"] = target
            st.session_state["scaler"] = scaler
            st.session_state["imputer"] = imp
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["le_target"] = le_target if task == "classification" else None
            st.session_state["X_enc_cols"] = X_enc.columns.tolist()

        # ── Results ──────────────────────────────────────────────────────────
        results = st.session_state["results"]
        section("Model Leaderboard", "🏆")

        if task == "classification":
            metric_key = "Accuracy"
            res_df = pd.DataFrame([{"Model": k, "Accuracy": v["Accuracy"]} for k, v in results.items()])
            res_df = res_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
        else:
            metric_key = "R²"
            res_df = pd.DataFrame([{"Model": k, "R²": v["R²"], "MAE": v["MAE"], "RMSE": v["RMSE"]}
                                    for k, v in results.items()])
            res_df = res_df.sort_values("R²", ascending=False).reset_index(drop=True)

        st.dataframe(res_df, use_container_width=True, hide_index=True)

        best_model_name = res_df.iloc[0]["Model"]
        best_score = res_df.iloc[0][metric_key]
        st.session_state["best_model_name"] = best_model_name

        st.markdown(f'<div class="info-box">🥇 Best model: <strong>{best_model_name}</strong> '
                    f'— {metric_key}: <strong>{best_score}</strong></div>', unsafe_allow_html=True)

        # Bar chart
        fig_lb = px.bar(res_df, x="Model", y=metric_key, template="plotly_dark",
                        color=metric_key, color_continuous_scale=["#6C63FF","#00FFB2"],
                        text_auto=".4f")
        apply_theme(fig_lb)
        fig_lb.update_layout(height=300, margin=dict(t=20,b=20), showlegend=False)
        st.plotly_chart(fig_lb, use_container_width=True)

        # Feature importance
        best_model = trained[best_model_name]
        if hasattr(best_model, "feature_importances_"):
            section("Feature Importance", "🔑")
            imp_df = pd.DataFrame({
                "Feature": features,
                "Importance": best_model.feature_importances_
            }).sort_values("Importance", ascending=True).tail(20)
            fig_fi = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                            template="plotly_dark", color="Importance",
                            color_continuous_scale=["#6C63FF","#00FFB2"])
            apply_theme(fig_fi)
            fig_fi.update_layout(height=max(250, len(imp_df)*22), margin=dict(t=10,b=10))
            st.plotly_chart(fig_fi, use_container_width=True)

        # Confusion matrix for classification
        if task == "classification":
            section("Confusion Matrix — Best Model", "🔲")
            y_pred_best = trained[best_model_name].predict(st.session_state["X_test"])
            cm = confusion_matrix(st.session_state["y_test"], y_pred_best)
            le_t = st.session_state["le_target"]
            labels = le_t.classes_.tolist() if le_t else list(range(cm.shape[0]))
            fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True,
                               color_continuous_scale=["#0A0E1A","#6C63FF","#00FFB2"],
                               template="plotly_dark",
                               labels=dict(x="Predicted", y="Actual"))
            apply_theme(fig_cm)
            fig_cm.update_layout(height=350, margin=dict(t=20,b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section("Live Prediction", "🔮")

    if "trained" not in st.session_state:
        st.markdown('<div class="warn-box">⚠️ Train models first in the <strong>ML Training</strong> tab.</div>',
                    unsafe_allow_html=True)
    else:
        results = st.session_state["results"]
        trained = st.session_state["trained"]
        task = st.session_state["task"]
        features = st.session_state["features"]
        scaler = st.session_state["scaler"]
        imp = st.session_state["imputer"]
        le_t = st.session_state["le_target"]
        best_name = st.session_state.get("best_model_name", list(trained.keys())[0])

        chosen_model = st.selectbox("Select Model", list(trained.keys()),
                                    index=list(trained.keys()).index(best_name))

        st.markdown(f'<div class="info-box">Enter values for each feature to get a real-time prediction from <strong>{chosen_model}</strong>.</div>',
                    unsafe_allow_html=True)

        input_vals = {}
        cols_per_row = 3
        feature_chunks = [features[i:i+cols_per_row] for i in range(0, len(features), cols_per_row)]

        for chunk in feature_chunks:
            cols = st.columns(len(chunk))
            for ci, feat in enumerate(chunk):
                col_data = df[feat]
                if col_data.dtype == object or col_data.nunique() <= 10:
                    opts = sorted(col_data.dropna().unique().tolist())
                    input_vals[feat] = cols[ci].selectbox(feat, opts, key=f"pred_{feat}")
                else:
                    mn, mx, med = float(col_data.min()), float(col_data.max()), float(col_data.median())
                    input_vals[feat] = cols[ci].number_input(feat, min_value=mn, max_value=mx, value=med, key=f"pred_{feat}")

        pred_btn = st.button("🔮 Predict Now")

        if pred_btn:
            row = pd.DataFrame([input_vals])
            row_enc, _ = encode_df(row)
            # align columns
            for c in st.session_state["X_enc_cols"]:
                if c not in row_enc.columns:
                    row_enc[c] = 0
            row_enc = row_enc[st.session_state["X_enc_cols"]]
            row_imp = imp.transform(row_enc)
            row_sc = scaler.transform(row_imp)
            prediction = trained[chosen_model].predict(row_sc)[0]

            if task == "classification" and le_t is not None:
                pred_label = le_t.inverse_transform([int(prediction)])[0]
            else:
                pred_label = round(float(prediction), 4)

            # Confidence for classifiers
            conf_str = ""
            model_obj = trained[chosen_model]
            if hasattr(model_obj, "predict_proba"):
                proba = model_obj.predict_proba(row_sc)[0]
                confidence = round(max(proba) * 100, 1)
                conf_str = f" &nbsp;|&nbsp; Confidence: <strong>{confidence}%</strong>"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(0,255,178,0.1),rgba(108,99,255,0.1));
                        border:1px solid rgba(0,255,178,0.3); border-radius:14px;
                        padding:2rem; text-align:center; margin-top:1rem;">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#7C8DB5;
                            text-transform:uppercase;letter-spacing:2px;">Prediction Result</div>
                <div style="font-family:'Space Mono',monospace;font-size:2.8rem;
                            color:#00FFB2;font-weight:700;margin:.5rem 0;">{pred_label}</div>
                <div style="font-size:0.85rem;color:#7C8DB5;">Model: <strong style="color:#6C63FF">{chosen_model}</strong>{conf_str}</div>
            </div>
            """, unsafe_allow_html=True)

            if task == "classification" and hasattr(model_obj, "predict_proba"):
                proba = model_obj.predict_proba(row_sc)[0]
                classes = le_t.classes_ if le_t else list(range(len(proba)))
                fig_prob = px.bar(x=[str(c) for c in classes], y=proba,
                                  labels={"x": "Class", "y": "Probability"},
                                  template="plotly_dark", color=proba,
                                  color_continuous_scale=["#6C63FF","#00FFB2"])
                apply_theme(fig_prob)
                fig_prob.update_layout(height=260, margin=dict(t=20,b=20), showlegend=False,
                                       title="Class Probabilities")
                st.plotly_chart(fig_prob, use_container_width=True)
