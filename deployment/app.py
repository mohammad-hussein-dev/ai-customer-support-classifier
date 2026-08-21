"""Streamlit web application for ticket classification.

Provides an interactive interface for manual ticket classification
and batch processing with confidence thresholds.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st

from src.models.predict_model import TicketClassifier


def main():
    """Run the Streamlit ticket classification app."""
    st.set_page_config(
        page_title="AI Ticket Classifier",
        page_icon="🎫",
        layout="wide",
    )

    st.title("🎫 AI Customer Support Ticket Classifier")
    st.markdown("""
    Automatically classify customer support tickets into:
    **Billing**, **Technical Support**, **Account**, and **Refund**.
    """)

    # Sidebar
    st.sidebar.header("Configuration")
    review_threshold = st.sidebar.slider(
        "Review Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Tickets below this confidence are flagged for human review.",
    )

    @st.cache_resource
    def load_classifier(threshold: float):
        return TicketClassifier.from_directory(
            "models/production",
            review_threshold=threshold,
        )

    try:
        classifier = load_classifier(review_threshold)
        st.sidebar.success("✅ Model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["Single Ticket", "Batch Upload"])

    with tab1:
        st.subheader("Classify a Single Ticket")
        ticket_text = st.text_area(
            "Enter ticket text:",
            height=150,
            placeholder="Example: I was charged twice for my subscription...",
        )

        if st.button("Classify", type="primary") and ticket_text:
            with st.spinner("Analyzing..."):
                result = classifier.predict(ticket_text)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Category", result["category"])
            with col2:
                st.metric("Confidence", f"{result['confidence']:.2%}")
            with col3:
                status = "⚠️ Needs Review" if result["needs_review"] else "✅ Auto-route"
                st.metric("Status", status)

            st.subheader("Category Probabilities")
            probs = result["all_probabilities"]
            for cat, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                st.progress(prob, text=f"{cat}: {prob:.2%}")

    with tab2:
        st.subheader("Batch Classification")
        uploaded_file = st.file_uploader("Upload CSV with 'text' column", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column")
            else:
                if st.button("Process Batch"):
                    progress_bar = st.progress(0)
                    results = []

                    for i, text in enumerate(df["text"]):
                        results.append(classifier.predict(str(text)))
                        progress_bar.progress((i + 1) / len(df))

                    df["predicted_category"] = [r["category"] for r in results]
                    df["confidence"] = [r["confidence"] for r in results]
                    df["needs_review"] = [r["needs_review"] for r in results]

                    st.dataframe(df)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results",
                        csv,
                        "classified_tickets.csv",
                        "text/csv",
                    )


if __name__ == "__main__":
    main()
