#!/usr/bin/env python3
"""
Train and Evaluate 3 ML Models on Real Hybrid Dataset.
Produces realistic F1 scores (not synthetic 1.00).
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_data() -> tuple:
    """Load preprocessed train/test splits."""
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df["text_clean"].fillna("").astype(str)
    X_test = test_df["text_clean"].fillna("").astype(str)
    y_train = train_df["category"].values
    y_test = test_df["category"].values
    
    print(f"📥 Data loaded:")
    print(f"   🚂 Train: {len(X_train)} samples")
    print(f"   🧪 Test:  {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def build_features(X_train, X_test) -> tuple:
    """Build TF-IDF features with optimal hyperparameters."""
    print("\n🔧 Building TF-IDF features...")
    
    vectorizer = TfidfVectorizer(
        max_features=3000,           # Increased for real data
        ngram_range=(1, 2),          # Unigrams + bigrams
        sublinear_tf=True,           # Log scaling
        min_df=3,                    # Ignore very rare terms
        max_df=0.90,                 # Ignore very common terms
        stop_words="english",
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"   ✅ Vocabulary size: {vocab_size}")
    print(f"   📊 Train shape: {X_train_vec.shape}")
    print(f"   📊 Test shape:  {X_test_vec.shape}")
    
    return vectorizer, X_train_vec, X_test_vec


def encode_labels(y_train, y_test) -> tuple:
    """Encode string labels to integers."""
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    
    print(f"\n🏷️  Label encoding:")
    for i, cls in enumerate(encoder.classes_):
        print(f"      {i}: {cls}")
    
    return encoder, y_train_enc, y_test_enc


def train_and_evaluate_models(
    X_train, X_test, y_train, y_test, encoder
) -> dict:
    """
    Train 3 models and compare with cross-validation.
    Returns dict of results for each model.
    """
    print("\n" + "=" * 60)
    print("🤖 Training and Evaluating Models")
    print("=" * 60)
    
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",  # Handle imbalance
            random_state=42,
            n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",
            probability=True,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
    }
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"📦 {name}")
        print(f"{'─' * 50}")
        
        # Cross-validation
        print("   🔄 5-Fold Cross-Validation...")
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="f1_macro", n_jobs=-1
        )
        print(f"      CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        print("   🏋️ Training on full train set...")
        model.fit(X_train, y_train)
        
        # Test evaluation
        print("   🧪 Evaluating on test set...")
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        
        print(f"      ✅ Accuracy:  {acc:.4f}")
        print(f"      ✅ F1 Macro:  {f1_macro:.4f}")
        print(f"      ✅ F1 Weighted: {f1_weighted:.4f}")
        
        # Per-class F1
        print("   📊 Per-Class F1:")
        report = classification_report(
            y_test, y_pred,
            target_names=encoder.classes_,
            output_dict=True
        )
        for cls in encoder.classes_:
            f1 = report[cls]["f1-score"]
            support = int(report[cls]["support"])
            print(f"      {cls:20s}: {f1:.4f} (n={support})")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("   🎯 Confusion Matrix:")
        print(f"      {encoder.classes_}")
        print(cm)
        
        # Store results
        results[name] = {
            "model": model,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "test_accuracy": acc,
            "test_f1_macro": f1_macro,
            "test_f1_weighted": f1_weighted,
            "per_class_f1": {cls: report[cls]["f1-score"] for cls in encoder.classes_},
            "confusion_matrix": cm.tolist(),
        }
    
    return results


def select_best_model(results: dict) -> tuple:
    """Select best model based on test F1-macro."""
    print("\n" + "=" * 60)
    print("🏆 Model Comparison and Selection")
    print("=" * 60)
    
    # Sort by test F1-macro
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]["test_f1_macro"],
        reverse=True
    )
    
    print("\n   📊 Leaderboard (by Test F1-Macro):")
    print(f"      {'Rank':<6} {'Model':<20} {'CV F1':<12} {'Test F1':<12} {'Acc':<10}")
    print(f"      {'-' * 60}")
    for rank, (name, res) in enumerate(sorted_models, 1):
        print(f"      {rank:<6} {name:<20} {res['cv_f1_mean']:.4f}       {res['test_f1_macro']:.4f}       {res['test_accuracy']:.4f}")
    
    best_name, best_res = sorted_models[0]
    print(f"\n   🥇 BEST MODEL: {best_name}")
    print(f"      Test F1-Macro: {best_res['test_f1_macro']:.4f}")
    print(f"      Test Accuracy: {best_res['test_accuracy']:.4f}")
    
    return best_name, best_res


def save_artifacts(
    best_name: str,
    best_res: dict,
    vectorizer,
    encoder,
    results: dict
) -> None:
    """Save model, vectorizer, encoder, and metrics."""
    output_dir = Path("models/production_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best model
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(best_res["model"], f)
    
    # Save vectorizer
    with open(output_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save encoder
    with open(output_dir / "encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    
    # Save metrics as JSON
    metrics = {
        "best_model": best_name,
        "models": {
            name: {
                "cv_f1_mean": res["cv_f1_mean"],
                "cv_f1_std": res["cv_f1_std"],
                "test_accuracy": res["test_accuracy"],
                "test_f1_macro": res["test_f1_macro"],
                "test_f1_weighted": res["test_f1_weighted"],
                "per_class_f1": res["per_class_f1"],
            }
            for name, res in results.items()
        }
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Artifacts saved to {output_dir}/")
    print("   📦 model.pkl (best classifier)")
    print("   📦 vectorizer.pkl (TF-IDF)")
    print("   📦 encoder.pkl (label mapping)")
    print("   📄 metrics.json (all results)")


def main() -> None:
    """Main pipeline: features → train → evaluate → save."""
    print("=" * 60)
    print("🔬 Feature Engineering + Model Training")
    print("=" * 60)
    
    # 1. Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Build features
    vectorizer, X_train_vec, X_test_vec = build_features(X_train, X_test)
    
    # 3. Encode labels
    encoder, y_train_enc, y_test_enc = encode_labels(y_train, y_test)
    
    # 4. Train and evaluate
    results = train_and_evaluate_models(
        X_train_vec, X_test_vec, y_train_enc, y_test_enc, encoder
    )
    
    # 5. Select best
    best_name, best_res = select_best_model(results)
    
    # 6. Save everything
    save_artifacts(best_name, best_res, vectorizer, encoder, results)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"   🥇 Best Model: {best_name}")
    print(f"   📊 Test F1-Macro: {best_res['test_f1_macro']:.4f}")
    print(f"   📊 Test Accuracy: {best_res['test_accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
