#!/usr/bin/env python3
"""
Error Analysis and Model Explainability with SHAP.
Identifies misclassification patterns and explains predictions.
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_artifacts():
    """Load model, vectorizer, encoder, and test data."""
    model_dir = Path("models/production_v2")
    
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir / "encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df["text_clean"].fillna("").astype(str)
    y_test = test_df["category"].values
    
    return model, vectorizer, encoder, X_test, y_test


def analyze_errors(model, vectorizer, encoder, X_test, y_test):
    """Find and analyze misclassified samples."""
    print("=" * 60)
    print("🔍 Error Analysis: Top Misclassifications")
    print("=" * 60)
    
    X_vec = vectorizer.transform(X_test)
    y_pred = encoder.inverse_transform(model.predict(X_vec))
    
    # Find misclassified
    errors = []
    for i, (true, pred, text) in enumerate(zip(y_test, y_pred, X_test)):
        if true != pred:
            errors.append({
                "index": i,
                "true": true,
                "predicted": pred,
                "text": text,
            })
    
    print(f"\n   ❌ Total misclassified: {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")
    
    # Confusion pairs
    pairs = Counter((e["true"], e["predicted"]) for e in errors)
    print("\n   📊 Top Confused Pairs (True → Predicted):")
    for (true, pred), count in pairs.most_common(10):
        print(f"      {true:20s} → {pred:20s}: {count} cases")
    
    # Show examples of top errors
    print("\n   📝 Example Misclassifications:")
    for (true, pred), count in pairs.most_common(5):
        examples = [e for e in errors if e["true"] == true and e["predicted"] == pred][:2]
        print(f"\n      {true} → {pred} ({count} cases):")
        for ex in examples:
            print(f"         \"{ex['text'][:100]}...\"")
    
    return errors, pairs


def feature_importance_analysis(model, vectorizer, encoder):
    """Analyze which features (words) are most important per class."""
    print("\n" + "=" * 60)
    print("🔑 Feature Importance Analysis")
    print("=" * 60)
    
    feature_names = vectorizer.get_feature_names_out()
    classes = encoder.classes_
    
    # For LogisticRegression: coef_ per class
    if hasattr(model, "coef_"):
        print("\n   📊 Top Features per Class (Logistic Regression):")
        for i, cls in enumerate(classes):
            coef = model.coef_[i]
            top_idx = np.argsort(coef)[-15:][::-1]
            top_words = [(feature_names[j], coef[j]) for j in top_idx]
            print(f"\n      {cls}:")
            for word, score in top_words:
                print(f"         {word:20s}: {score:+.4f}")
    
    # For SVM: similar if linear, otherwise skip
    elif hasattr(model, "support_vectors_"):
        print("\n   ℹ️  SVM (RBF) — feature importance not directly available.")
        print("      Using SHAP for explainability instead.")
    
    # For RandomForest: feature_importances_
    elif hasattr(model, "feature_importances_"):
        print("\n   📊 Top Global Features (Random Forest):")
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-20:][::-1]
        for j in top_idx:
            print(f"      {feature_names[j]:20s}: {importances[j]:.4f}")


def shap_analysis(model, vectorizer, encoder, X_test, y_test, sample_size=100):
    """SHAP explainability for sample predictions."""
    print("\n" + "=" * 60)
    print("🎨 SHAP Explainability")
    print("=" * 60)
    
    try:
        import shap
        
        # Sample for speed
        sample_idx = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
        X_sample = X_test.iloc[sample_idx] if hasattr(X_test, "iloc") else X_test[sample_idx]
        y_sample = y_test[sample_idx]
        
        X_vec = vectorizer.transform(X_sample)
        
        # SHAP for tree models (RandomForest)
        if hasattr(model, "estimators_"):
            print("   🌲 Using TreeSHAP for RandomForest...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_vec.toarray())
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_vec.toarray(),
                feature_names=vectorizer.get_feature_names_out(),
                show=False
            )
            plt.tight_layout()
            Path("reports/figures").mkdir(parents=True, exist_ok=True)
            plt.savefig("reports/figures/shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("   ✅ Saved: reports/figures/shap_summary.png")
        
        # SHAP for linear models (LogisticRegression)
        elif hasattr(model, "coef_"):
            print("   📈 Using LinearSHAP for LogisticRegression...")
            explainer = shap.LinearExplainer(model, X_vec)
            shap_values = explainer.shap_values(X_vec.toarray())
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_vec.toarray(),
                feature_names=vectorizer.get_feature_names_out(),
                show=False
            )
            plt.tight_layout()
            Path("reports/figures").mkdir(parents=True, exist_ok=True)
            plt.savefig("reports/figures/shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("   ✅ Saved: reports/figures/shap_summary.png")
        
        # For SVM: use KernelSHAP (slow, sample smaller)
        else:
            print("   🎯 Using KernelSHAP for SVM (sample=50)...")
            X_small = X_vec[:50]
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_small, 10))
            shap_values = explainer.shap_values(X_small.toarray(), nsamples=50)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_small.toarray(),
                feature_names=vectorizer.get_feature_names_out(),
                show=False
            )
            plt.tight_layout()
            plt.savefig("reports/figures/shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("   ✅ Saved: reports/figures/shap_summary.png")
        
        # Explain single prediction
        print("\n   🔍 Example Explanation:")
        idx = 0
        text = X_sample.iloc[idx] if hasattr(X_sample, "iloc") else X_sample[idx]
        true_label = y_sample[idx]
        pred_label = encoder.inverse_transform(model.predict(X_vec[idx]))[0]
        
        print(f"      Text: \"{text[:80]}...\"")
        print(f"      True: {true_label}")
        print(f"      Predicted: {pred_label}")
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_vec[idx])[0]
            print(f"      Confidence: {np.max(proba):.2%}")
            print(f"      Top 3 classes:")
            top3 = np.argsort(proba)[-3:][::-1]
            for j in top3:
                print(f"         {encoder.classes_[j]:20s}: {proba[j]:.2%}")
        
    except ImportError:
        print("   ⚠️  SHAP not installed. Skipping.")
        print("      Run: pip install shap")
    except Exception as e:
        print(f"   ⚠️  SHAP error: {e}")
        print("      This is normal for some model types.")


def generate_insights_report(errors, pairs, model, encoder):
    """Generate actionable insights for business."""
    print("\n" + "=" * 60)
    print("💡 Business Insights")
    print("=" * 60)
    
    total_errors = len(errors)
    total_samples = sum(pairs.values()) + total_errors  # Approximate
    
    insights = {
        "total_test_samples": int(total_samples),
        "total_errors": total_errors,
        "error_rate": round(total_errors / total_samples, 4),
        "top_confusion_pairs": [
            {"true": t, "predicted": p, "count": c}
            for (t, p), c in pairs.most_common(5)
        ],
        "recommendations": []
    }
    
    # Generate recommendations based on errors
    top_pair = pairs.most_common(1)
    if top_pair:
        (true, pred), count = top_pair[0]
        insights["recommendations"].append(
            f"Review {true} vs {pred} boundary: {count} misclassifications. "
            f"Consider adding more training data for these classes."
        )
    
    # Refund analysis
    refund_errors = [e for e in errors if e["true"] == "Refund"]
    if refund_errors:
        insights["recommendations"].append(
            f"Refund class has {len(refund_errors)} errors. "
            f"Consider human review for all Refund tickets (low volume, high risk)."
        )
    
    # Save report
    report_dir = Path("reports/tables")
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "error_analysis.json", "w") as f:
        json.dump(insights, f, indent=2)
    
    print(f"\n   📄 Saved: reports/tables/error_analysis.json")
    print("\n   🎯 Key Recommendations:")
    for rec in insights["recommendations"]:
        print(f"      • {rec}")
    
    print(f"\n   📊 Error Rate: {insights['error_rate']*100:.1f}%")
    print(f"   ✅ Model Accuracy: {(1-insights['error_rate'])*100:.1f}%")


def main():
    """Main pipeline: errors → features → SHAP → insights."""
    print("=" * 60)
    print("🔬 Error Analysis + Explainability")
    print("=" * 60)
    
    # Load everything
    print("\n📦 Loading artifacts...")
    model, vectorizer, encoder, X_test, y_test = load_artifacts()
    print("   ✅ Loaded")
    
    # 1. Error analysis
    errors, pairs = analyze_errors(model, vectorizer, encoder, X_test, y_test)
    
    # 2. Feature importance
    feature_importance_analysis(model, vectorizer, encoder)
    
    # 3. SHAP
    shap_analysis(model, vectorizer, encoder, X_test, y_test)
    
    # 4. Business insights
    generate_insights_report(errors, pairs, model, encoder)
    
    print("\n" + "=" * 60)
    print("✅ Error Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
