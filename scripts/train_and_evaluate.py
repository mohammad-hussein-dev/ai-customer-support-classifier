#!/usr/bin/env python3
"""
End-to-End Training & Evaluation Pipeline
==========================================
Professional pipeline with Rich terminal output, dark-theme figures,
and comprehensive error analysis for banking intent classification.

Usage:
    python scripts/train_and_evaluate.py
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import TextPreprocessor
from src.models.evaluate_model import ModelEvaluator
from src.models.train_model import ModelTrainer
from src.utils.metrics import MetricsPrinter, timer, print_pipeline_stage, print_success, print_file_saved
from src.visualization.visualize import BankingVisualizer

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SELECTED_INTENTS = [
    "card_arrival", "card_not_working", "cash_withdrawal_not_recognised",
    "declined_card_payment", "lost_or_stolen_card", "transaction_charged_twice",
    "transfer_not_received_by_recipient", "cash_withdrawal_charge",
]

BUSINESS_CRITICAL = [
    "lost_or_stolen_card", "declined_card_payment", "transaction_charged_twice",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
REVIEW_THRESHOLD = 0.7


def load_data(data_path: str = "data/processed") -> pd.DataFrame:
    """Load and filter banking dataset to selected intents."""
    print_pipeline_stage(1, 5, "Data Loading", "Loading and filtering Banking77 dataset")
    
    paths_to_try = [
        Path(data_path) / "hybrid_dataset.csv",
        Path(data_path).parent / "raw" / "hybrid_dataset.csv",
        Path("data/raw/hybrid_dataset.csv"),
        Path("data/processed/train.csv"),
    ]
    
    df = None
    for p in paths_to_try:
        if p.exists():
            df = pd.read_csv(p)
            console.print(f"[dim]Loaded: {p} ({len(df):,} rows)[/dim]")
            break
    
    if df is None:
        console.print("[yellow]⚠️  No dataset found. Creating demo dataset...[/yellow]")
        df = create_demo_data()
    
    if "intent" in df.columns or "category" in df.columns:
        label_col = "intent" if "intent" in df.columns else "category"
        df = df[df[label_col].isin(SELECTED_INTENTS)].copy()
        console.print(f"[green]✅ Filtered to {len(SELECTED_INTENTS)} intents: {len(df):,} samples[/green]")
    
    return df


def create_demo_data() -> pd.DataFrame:
    """Create minimal demo data for testing the pipeline."""
    data = []
    templates = {
        "card_arrival": [
            "My card has not arrived yet",
            "When will my new card be delivered",
            "I ordered a card two weeks ago still waiting",
        ],
        "card_not_working": [
            "My card is not working at the ATM",
            "The card stopped working today",
            "I cannot use my card anywhere",
        ],
        "cash_withdrawal_not_recognised": [
            "I did not make this withdrawal",
            "Unknown cash withdrawal on my account",
            "This ATM withdrawal is not mine",
        ],
        "declined_card_payment": [
            "My payment was declined at the store",
            "Card declined when I tried to buy groceries",
            "Why was my card payment rejected",
        ],
        "lost_or_stolen_card": [
            "I lost my card please block it",
            "My card was stolen I need help",
            "Someone took my card need to cancel",
        ],
        "transaction_charged_twice": [
            "I was charged twice for the same purchase",
            "Duplicate transaction on my statement",
            "Why is there a double charge",
        ],
        "transfer_not_received_by_recipient": [
            "The recipient did not receive my transfer",
            "My bank transfer never arrived",
            "Where is the money I sent",
        ],
        "cash_withdrawal_charge": [
            "Why was I charged for this ATM withdrawal",
            "Extra fee on cash withdrawal",
            "I was charged for withdrawing my own money",
        ],
    }
    
    for intent, texts in templates.items():
        for text in texts:
            data.append({"text": text, "intent": intent})
    
    df = pd.DataFrame(data)
    df = pd.concat([df] * 20, ignore_index=True)
    return df


def build_keyword_baseline() -> Dict[str, str]:
    """Build a simple keyword-based baseline classifier."""
    print_pipeline_stage(3, 5, "Keyword Baseline", "Rule-based classifier for comparison")
    
    rules = {
        "card_arrival": ["arrive", "deliver", "not arrived", "waiting for card", "card delivery"],
        "card_not_working": ["not working", "stopped working", "cannot use", "card broken", "card issue"],
        "cash_withdrawal_not_recognised": ["did not make", "not mine", "unknown withdrawal", "unrecognized", "fraud"],
        "declined_card_payment": ["declined", "rejected", "payment failed", "not accepted", "transaction declined"],
        "lost_or_stolen_card": ["lost", "stolen", "block", "cancel", "someone took"],
        "transaction_charged_twice": ["twice", "double charge", "duplicate", "charged two", "same purchase twice"],
        "transfer_not_received_by_recipient": ["not received", "never arrived", "where is", "recipient", "transfer"],
        "cash_withdrawal_charge": ["charged for withdrawal", "atm fee", "withdrawal charge", "cash fee"],
    }
    
    def predict(text: str) -> str:
        text_lower = text.lower()
        scores = {}
        for intent, keywords in rules.items():
            scores[intent] = sum(1 for kw in keywords if kw in text_lower)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "card_arrival"
    
    return predict


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate banking intent classifier")
    parser.add_argument("--data", default="data/processed", help="Data directory")
    parser.add_argument("--output", default="models/production_v2", help="Model output directory")
    parser.add_argument("--reports", default="reports", help="Reports output directory")
    args = parser.parse_args()
    
    console.print()
    console.rule("[bold cyan]🏦 BANKING INTENT CLASSIFIER — TRAINING PIPELINE", style="cyan")
    console.print()
    
    df = load_data(args.data)
    label_col = "intent" if "intent" in df.columns else "category"
    text_col = "text"
    
    print_pipeline_stage(2, 5, "Preprocessing", "Text cleaning, tokenization, lemmatization")
    with timer("Text Preprocessing"):
        preprocessor = TextPreprocessor(
            lowercase=True, remove_punctuation=True, remove_stopwords=True,
            lemmatize=True, min_word_length=2,
        )
        df["clean_text"] = preprocessor.transform(df[text_col].tolist())
        print_success(f"Preprocessed {len(df)} documents")
    
    baseline_predict = build_keyword_baseline()
    baseline_preds = [baseline_predict(t) for t in df[text_col]]
    baseline_f1 = f1_score(df[label_col], baseline_preds, average="macro", zero_division=0)
    console.print(f"[green]✅ Keyword Baseline Macro-F1: {baseline_f1:.4f}[/green]")
    console.print()
    
    print_pipeline_stage(4, 5, "Train/Test Split", "Stratified split with fixed random seed")
    X = df["clean_text"].values
    y = df[label_col].values
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y,
    )
    console.print(f"[dim]Train: {len(X_train):,} | Test: {len(X_test):,}[/dim]")
    
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    
    print_pipeline_stage(5, 5, "Feature Engineering & Training", "TF-IDF + Logistic Regression")
    with timer("TF-IDF Vectorization"):
        vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=2,
            max_df=0.95, sublinear_tf=True, strip_accents="unicode",
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        console.print(f"[dim]Vocabulary size: {len(vectorizer.vocabulary_):,}[/dim]")
    
    with timer("Model Training"):
        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X_train_tfidf, y_train_enc)
        print_success("Logistic Regression trained")
    
    y_pred_enc = model.predict(X_test_tfidf)
    y_pred = encoder.inverse_transform(y_pred_enc)
    y_proba = model.predict_proba(X_test_tfidf)
    
    console.print()
    console.rule("[bold green]📊 EVALUATION", style="green")
    console.print()
    
    evaluator = ModelEvaluator(
        class_names=encoder.classes_.tolist(),
        output_dir=args.reports,
        business_critical=BUSINESS_CRITICAL,
    )
    
    results = evaluator.evaluate(
        y_true=y_test, y_pred=y_pred, y_proba=y_proba,
        texts=[df[text_col].iloc[i] for i in idx_test],
        model=model, vectorizer=vectorizer,
    )
    
    test_f1 = results["overall"]["macro_f1"]
    evaluator.visualizer.plot_baseline_comparison(baseline_f1, test_f1)
    
    console.print("[dim]Generating EDA dashboard...[/dim]")
    viz = BankingVisualizer(output_dir=Path(args.reports) / "figures")
    viz.plot_eda_dashboard(df, text_col=text_col, label_col=label_col)
    
    console.print()
    console.rule("[bold blue]💾 SAVING ARTIFACTS", style="blue")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, output_dir / "model.pkl")
    joblib.dump(vectorizer, output_dir / "vectorizer.pkl")
    joblib.dump(encoder, output_dir / "encoder.pkl")
    joblib.dump(preprocessor, output_dir / "preprocessor.pkl")
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print_file_saved(str(output_dir / "model.pkl"), "Model")
    print_file_saved(str(output_dir / "vectorizer.pkl"), "Vectorizer")
    print_file_saved(str(output_dir / "encoder.pkl"), "Encoder")
    print_file_saved(str(output_dir / "metrics.json"), "Metrics")
    
    console.print()
    console.rule("[bold green]🎉 PIPELINE COMPLETE", style="green")
    console.print(f"[bold cyan]Final Macro-F1: {test_f1:.4f}[/bold cyan]")
    console.print(f"[dim]All artifacts saved to: {output_dir}[/dim]")
    console.print()


if __name__ == "__main__":
    main()
