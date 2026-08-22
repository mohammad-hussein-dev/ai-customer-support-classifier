"""
Professional Streamlit Dashboard — Banking Intent Classifier
============================================================
Dark theme with glassmorphism cards, interactive Plotly charts,
and real-time prediction with confidence visualization.

Run: streamlit run deployment/app.py
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Banking Intent Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM DARK THEME CSS (Glassmorphism)
# ═══════════════════════════════════════════════════════════════════════════════

DARK_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    }
    .glass-card {
        background: rgba(22, 27, 34, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(48, 54, 61, 0.6) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }
    h1, h2, h3 {
        color: #c9d1d9 !important;
        font-family: 'Segoe UI', system-ui, sans-serif !important;
    }
    h1 {
        background: linear-gradient(90deg, #58a6ff, #a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(35, 134, 54, 0.4) !important;
    }
    .stTextArea textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        color: #c9d1d9 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stMetric {
        background: rgba(22, 27, 34, 0.6) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        border: 1px solid rgba(48, 54, 61, 0.4) !important;
    }
    .stMetric label {
        color: #8b949e !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    .stMetric .css-1xarl3l {
        color: #58a6ff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(22, 27, 34, 0.6) !important;
        border-radius: 8px 8px 0 0 !important;
        color: #8b949e !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(88, 166, 255, 0.15) !important;
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }
    hr { border-color: #30363d !important; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg": "#0d1117", "card": "#161b22", "border": "#30363d",
    "text": "#c9d1d9", "text_dim": "#8b949e",
    "cyan": "#58a6ff", "green": "#3fb950", "yellow": "#d29922",
    "red": "#f85149", "purple": "#a371f7", "pink": "#f778ba",
}

INTENT_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#a371f7", "#f778ba", "#56d364", "#79c0ff"]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_artifacts():
    """Load model, vectorizer, and preprocessor."""
    model_dir = Path("models/production_v2")
    try:
        with open(model_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(model_dir / "vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open(model_dir / "encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        return model, vectorizer, encoder
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return None, None, None


@st.cache_data
def load_evaluation_data():
    """Load evaluation metrics and results."""
    try:
        with open("reports/tables/evaluation_results.json", "r") as f:
            return json.load(f)
    except:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_intent(text: str, model, vectorizer, encoder, threshold: float = 0.7) -> Dict[str, Any]:
    """Classify a banking support ticket with full explainability."""
    import re
    clean = re.sub(r"http\S+|www\S+|@\w+|\b\d+\b", "", text.lower())
    clean = re.sub(r"[^\w\s]", " ", clean)
    clean = " ".join(clean.split())
    
    features = vectorizer.transform([clean])
    pred_label = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    if hasattr(encoder, "inverse_transform"):
        pred_intent = encoder.inverse_transform([pred_label])[0]
        all_classes = encoder.classes_
    else:
        pred_intent = str(pred_label)
        all_classes = model.classes_
    
    confidence = float(np.max(probabilities))
    all_probs = {cls: float(p) for cls, p in zip(all_classes, probabilities)}
    
    feature_names = vectorizer.get_feature_names_out()
    top_features = []
    if hasattr(model, "coef_"):
        try:
            cls_idx = list(all_classes).index(pred_intent)
            coef = model.coef_[cls_idx]
            top_idx = np.argsort(coef)[-5:][::-1]
            top_features = [(feature_names[i], float(coef[i])) for i in top_idx if coef[i] > 0]
        except:
            pass
    
    return {
        "intent": pred_intent,
        "confidence": confidence,
        "needs_review": confidence < threshold,
        "all_probabilities": all_probs,
        "top_features": top_features,
        "preprocessed_text": clean,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_gauge_chart(confidence: float, threshold: float = 0.7) -> go.Figure:
    """Create a confidence gauge chart."""
    color = COLORS["green"] if confidence >= 0.9 else COLORS["yellow"] if confidence >= threshold else COLORS["red"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"suffix": "%", "font": {"size": 36, "color": color, "family": "Segoe UI"}},
        title={"text": "Confidence", "font": {"size": 14, "color": COLORS["text_dim"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["border"]},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": COLORS["card"],
            "borderwidth": 2,
            "bordercolor": COLORS["border"],
            "steps": [
                {"range": [0, threshold*100], "color": "rgba(248, 81, 73, 0.15)"},
                {"range": [threshold*100, 90], "color": "rgba(210, 153, 34, 0.15)"},
                {"range": [90, 100], "color": "rgba(63, 185, 80, 0.15)"},
            ],
            "threshold": {
                "line": {"color": COLORS["yellow"], "width": 3},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def render_probability_bars(probs: Dict[str, float]) -> go.Figure:
    """Create horizontal probability bars."""
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    labels = [p[0].replace("_", " ").title() for p in sorted_probs]
    values = [p[1] * 100 for p in sorted_probs]
    colors = [INTENT_COLORS[i % len(INTENT_COLORS)] for i in range(len(labels))]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels[::-1], x=values[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:.1f}%" for v in values[::-1]],
        textposition="outside",
        textfont={"color": COLORS["text"], "size": 11},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": "Probability (%)", "color": COLORS["text_dim"], "gridcolor": COLORS["border"]},
        yaxis={"color": COLORS["text"], "gridcolor": COLORS["border"]},
        height=max(300, len(labels) * 45),
        margin=dict(l=150, r=30, t=20, b=40),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; font-size: 24px;'>🏦 Banking AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8b949e; font-size: 12px;'>Intent Classification System</p>", unsafe_allow_html=True)
        st.divider()
        threshold = st.slider("Review Threshold", 0.0, 1.0, 0.7, 0.05,
                             help="Predictions below this confidence will be flagged for human review")
        st.divider()
        st.markdown("<p style='color: #8b949e; font-size: 11px;'>Built with ❤️ on Arch Linux</p>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; margin-bottom: 8px;'>AI Customer Support Ticket Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 32px;'>Automated intent detection for banking customer service</p>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📁 Batch Upload", "📊 Model Performance"])
    
    with tab1:
        col_input, col_result = st.columns([1.2, 1])
        with col_input:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Enter Customer Message")
            text_input = st.text_area("", placeholder="Example: I was charged twice for my card payment yesterday...", height=180, label_visibility="collapsed")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                predict_btn = st.button("🔮 Predict Intent", use_container_width=True)
            with col_btn2:
                clear_btn = st.button("🗑️ Clear", use_container_width=True)
            st.markdown("<p style='color: #8b949e; font-size: 11px; margin-top: 16px;'>Quick Examples:</p>", unsafe_allow_html=True)
            examples = [
                "My card hasn't arrived yet and I ordered it 2 weeks ago",
                "I tried to withdraw cash but the ATM charged me extra",
                "My card was declined at the grocery store today",
                "I think someone stole my card, I need to block it immediately",
            ]
            ex_cols = st.columns(2)
            for idx, example in enumerate(examples):
                with ex_cols[idx % 2]:
                    if st.button(example[:40] + "...", key=f"ex_{idx}", use_container_width=True):
                        st.session_state["text_input"] = example
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_result:
            if predict_btn and text_input.strip():
                model, vectorizer, encoder = load_model_artifacts()
                if model is not None:
                    result = predict_intent(text_input, model, vectorizer, encoder, threshold)
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.subheader("Prediction Result")
                    intent_color = INTENT_COLORS[0]
                    intent_display = result["intent"].replace("_", " ").title()
                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {intent_color}22, {intent_color}11);
                            border: 1px solid {intent_color}44; border-radius: 12px; padding: 16px; text-align: center; margin-bottom: 16px;">
                            <p style="color: {COLORS['text_dim']}; font-size: 12px; margin: 0;">Predicted Intent</p>
                            <h2 style="color: {intent_color}; margin: 4px 0; font-size: 28px;">{intent_display}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    if result["needs_review"]:
                        st.warning(f"⚠️ Low confidence ({result['confidence']:.1%}). Route to human agent.")
                    else:
                        st.success(f"✅ High confidence ({result['confidence']:.1%}). Auto-route approved.")
                    st.plotly_chart(render_gauge_chart(result["confidence"], threshold), use_container_width=True, config={"displayModeBar": False})
                    st.plotly_chart(render_probability_bars(result["all_probabilities"]), use_container_width=True, config={"displayModeBar": False})
                    if result["top_features"]:
                        st.markdown("<p style='color: #8b949e; font-size: 12px; margin-top: 16px;'>🔍 Top Contributing Terms:</p>", unsafe_allow_html=True)
                        feat_cols = st.columns(len(result["top_features"]))
                        for idx, (term, weight) in enumerate(result["top_features"]):
                            with feat_cols[idx]:
                                st.markdown(f"""
                                    <div style="background: {COLORS['card']}; border: 1px solid {COLORS['border']};
                                        border-radius: 8px; padding: 8px; text-align: center;">
                                        <p style="color: {COLORS['cyan']}; font-weight: bold; margin: 0;">{term}</p>
                                        <p style="color: {COLORS['text_dim']}; font-size: 10px; margin: 0;">{weight:.3f}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📁 Batch Classification")
        uploaded_file = st.file_uploader("Upload CSV with 'text' column", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column")
            else:
                model, vectorizer, encoder = load_model_artifacts()
                if model is not None:
                    with st.spinner("Processing..."):
                        results = []
                        for text in df["text"]:
                            results.append(predict_intent(str(text), model, vectorizer, encoder, threshold))
                        df["predicted_intent"] = [r["intent"] for r in results]
                        df["confidence"] = [r["confidence"] for r in results]
                        df["needs_review"] = [r["needs_review"] for r in results]
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1: st.metric("Total Processed", len(df))
                        with col_m2: st.metric("Auto-Routed", len(df) - df["needs_review"].sum())
                        with col_m3: st.metric("Needs Review", int(df["needs_review"].sum()))
                        intent_counts = df["predicted_intent"].value_counts()
                        fig = px.pie(values=intent_counts.values,
                            names=[n.replace("_", " ").title() for n in intent_counts.index],
                            color_discrete_sequence=INTENT_COLORS, hole=0.4)
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color=COLORS["text"], showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2), height=400)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        csv = df.to_csv(index=False)
                        st.download_button("💾 Download Results", csv, "predictions.csv", "text/csv")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        eval_data = load_evaluation_data()
        if eval_data:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📊 Model Performance Metrics")
            overall = eval_data.get("overall", {})
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1: st.metric("Accuracy", f"{overall.get('accuracy', 0):.4f}")
            with col_m2: st.metric("Macro-F1", f"{overall.get('macro_f1', 0):.4f}")
            with col_m3: st.metric("Weighted-F1", f"{overall.get('weighted_f1', 0):.4f}")
            per_class = eval_data.get("per_class", {})
            class_df = pd.DataFrame.from_dict(per_class, orient="index")
            class_df.index = [idx.replace("_", " ").title() for idx in class_df.index]
            class_df = class_df.round(4)
            st.dataframe(class_df, use_container_width=True)
            cm_path = Path("reports/figures/01_evaluation_dashboard.png")
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Run training first to generate evaluation metrics.")


if __name__ == "__main__":
    main()
