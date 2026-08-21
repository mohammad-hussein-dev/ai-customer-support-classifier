#!/usr/bin/env python3
"""
Streamlit Web App for AI Customer Support Ticket Classifier v1.0.
Compatible with production_v2 artifacts (SVM + TF-IDF + LabelEncoder).
"""

import sys
import pickle
import numpy as np
from pathlib import Path

import streamlit as st

# ═══════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Ticket Classifier v1.0",
    page_icon="🎫",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════
# Load Artifacts (cached)
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, vectorizer, and encoder from production_v2."""
    model_dir = Path("models/production_v2")
    
    if not model_dir.exists():
        st.error(f"❌ Model directory not found: {model_dir}")
        st.info("Please run: python3 scripts/train_and_evaluate.py")
        return None, None, None, None
    
    try:
        with open(model_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(model_dir / "vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open(model_dir / "encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open(model_dir / "metrics.json", "r") as f:
            import json
            metrics = json.load(f)
        
        return model, vectorizer, encoder, metrics
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None


def predict_ticket(text: str, model, vectorizer, encoder):
    """
    Predict category and confidence for a single ticket.
    
    Args:
        text: Raw customer ticket text
        model: Trained classifier
        vectorizer: TF-IDF vectorizer
        encoder: Label encoder
    
    Returns:
        dict with category, confidence, all_probabilities
    """
    # Vectorize
    X = vectorizer.transform([text])
    
    # Predict
    pred_idx = model.predict(X)[0]
    category = encoder.inverse_transform([pred_idx])[0]
    
    # Probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        all_proba = {
            cls: float(p) for cls, p in zip(encoder.classes_, proba)
        }
    else:
        confidence = 1.0
        all_proba = {category: 1.0}
    
    # Sort probabilities descending
    all_proba = dict(sorted(all_proba.items(), key=lambda x: x[1], reverse=True))
    
    return {
        "category": category,
        "confidence": confidence,
        "all_probabilities": all_proba
    }


# ═══════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════
def main():
    # Load artifacts
    model, vectorizer, encoder, metrics = load_artifacts()
    
    if model is None:
        st.stop()
    
    best_model = metrics.get("best_model", "Unknown") if metrics else "Unknown"
    test_f1 = metrics["models"][best_model]["test_f1_macro"] if metrics else 0.0
    test_acc = metrics["models"][best_model]["test_accuracy"] if metrics else 0.0
    
    # Header
    st.title("🎫 AI Customer Support Classifier")
    st.markdown(
        f"<p style='color: gray; font-size: 14px;'>"
        f"Model: <b>{best_model}</b> | "
        f"F1: <b>{test_f1:.3f}</b> | "
        f"Accuracy: <b>{test_acc:.1%}</b> | "
        f"v1.0</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Input
    text = st.text_area(
        "📝 Enter customer support ticket:",
        height=120,
        placeholder="Example: I was charged twice for my subscription this month...",
        help="Type or paste the customer ticket text here"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
    with col2:
        st.caption("Press Classify or hit Ctrl+Enter")
    
    # Classification
    if classify_btn or (text and text.strip() and st.session_state.get("auto_classify")):
        if not text or not text.strip():
            st.warning("⚠️ Please enter some text first.")
            return
        
        with st.spinner("Analyzing ticket..."):
            result = predict_ticket(text.strip(), model, vectorizer, encoder)
        
        category = result["category"]
        confidence = result["confidence"]
        all_proba = result["all_probabilities"]
        
        # Color coding by category
        colors = {
            "Account": "#4A90E2",
            "Billing": "#FFB800",
            "Technical Support": "#00D9A3",
            "Refund": "#FF4757"
        }
        color = colors.get(category, "#888888")
        
        # Result display
        st.markdown("---")
        st.subheader("📊 Classification Result")
        
        # Main result box
        st.markdown(
            f"<div style='background-color: {color}22; border-left: 5px solid {color}; "
            f"padding: 15px; border-radius: 8px; margin: 10px 0;'>"
            f"<h3 style='color: {color}; margin: 0;'>🏷️ {category}</h3>"
            f"<p style='margin: 5px 0 0 0; font-size: 18px;'>"
            f"Confidence: <b>{confidence:.1%}</b></p>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Progress bar
        st.progress(confidence)
        
        # Warning for low confidence
        if confidence < 0.70:
            st.warning(
                "⚠️ **Low confidence** — This ticket should be reviewed by a human agent."
            )
        elif confidence < 0.85:
            st.info(
                "ℹ️ **Medium confidence** — Consider double-checking this classification."
            )
        else:
            st.success("✅ **High confidence** — Classification is reliable.")
        
        # All probabilities
        st.markdown("#### 📈 All Class Probabilities")
        prob_cols = st.columns(len(all_proba))
        for i, (cls, prob) in enumerate(all_proba.items()):
            with prob_cols[i]:
                is_top = (cls == category)
                border = f"2px solid {colors.get(cls, '#888')}" if is_top else "1px solid #ddd"
                bg = f"{colors.get(cls, '#888')}11" if is_top else "#f8f9fa"
                st.markdown(
                    f"<div style='text-align: center; padding: 10px; border: {border}; "
                    f"border-radius: 8px; background-color: {bg};'>"
                    f"<div style='font-size: 24px; font-weight: bold; color: {colors.get(cls, '#888')};'>"
                    f"{prob:.1%}</div>"
                    f"<div style='font-size: 12px; color: #666;'>{cls}</div>"
                    f"{'<div style=\\\"font-size: 10px; color: green;\\\">✓ Top</div>' if is_top else ''}"
                    f"</div>",
                    unsafe_allow_html=True
                )
        
        # Explanation
        st.markdown("---")
        st.subheader("🔑 Key Words")
        
        # Show top TF-IDF features for this text
        X = vectorizer.transform([text.strip()])
        feature_names = vectorizer.get_feature_names_out()
        scores = X.toarray()[0]
        top_idx = np.argsort(scores)[-8:][::-1]
        
        words_html = ""
        for idx in top_idx:
            if scores[idx] > 0:
                word = feature_names[idx]
                score = scores[idx]
                words_html += (
                    f"<span style='display: inline-block; background: #f0f0f0; "
                    f"padding: 4px 10px; margin: 3px; border-radius: 12px; "
                    f"font-size: 13px;'>"
                    f"{word} <span style='color: #888;'>({score:.3f})</span>"
                    f"</span>"
                )
        
        st.markdown(words_html, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption(
        "🤖 Powered by SVM + TF-IDF | "
        "📊 Trained on 13,783 real banking tickets | "
        "🐙 [GitHub](https://github.com/mohammad-hussein-dev/ai-customer-support-classifier)"
    )


if __name__ == "__main__":
    main()
