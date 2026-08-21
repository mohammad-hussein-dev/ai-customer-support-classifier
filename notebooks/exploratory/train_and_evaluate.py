#!/usr/bin/env python3
"""Train models and evaluate performance."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessing import TextPreprocessor
from src.features.build_features import FeatureBuilder
from src.models.evaluate_model import ModelEvaluator
from src.models.train_model import ModelTrainer


def main() -> None:
    """Run training and evaluation pipeline."""
    print("=" * 70)
    print("🧠 TRAINING & EVALUATION — Customer Support Ticket Classifier")
    print("=" * 70)

    # Load and prepare data
    print("\n[1/5] Loading & preprocessing data...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "tickets.csv")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["category"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["category"],
    )

    preprocessor = TextPreprocessor()
    X_train_clean = preprocessor.transform(X_train_raw)
    X_test_clean = preprocessor.transform(X_test_raw)

    builder = FeatureBuilder(max_features=10000, ngram_range=(1, 2))
    X_train = builder.fit_transform(X_train_clean)
    X_test = builder.transform(X_test_clean)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"      ✓ Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"      ✓ Classes: {sorted(df['category'].unique())}")

    # Model configs
    models = [
        ("logistic_regression", {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
        }),
        ("svm", {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 2000,
        }),
        ("naive_bayes", {
            "alpha": 0.5,
        }),
    ]

    best_f1 = 0.0
    best_model_name = ""
    results_table = []

    print("\n[2/5] Cross-validating models...")
    for name, params in models:
        print(f"\n      → {name.upper()}")
        trainer = ModelTrainer(model_name=name, model_params=params, cv_folds=5)
        cv = trainer.cross_validate(X_train, y_train)
        f1_mean = np.mean(cv["test_f1_weighted"])
        f1_std = np.std(cv["test_f1_weighted"])
        print(f"        CV F1 (weighted): {f1_mean:.4f} (+/- {f1_std:.4f})")

    print("\n[3/5] Fitting & evaluating on test set...")
    for name, params in models:
        print(f"\n      → {name}")
        trainer = ModelTrainer(model_name=name, model_params=params)
        trainer.fit(X_train, y_train)

        y_pred = trainer.predict(X_test)
        y_proba = trainer.predict_proba(X_test)

        evaluator = ModelEvaluator(class_names=sorted(df["category"].unique()))
        results = evaluator.evaluate(y_test, y_pred, y_proba)

        results_table.append({
            "Model": name,
            "Accuracy": results["accuracy"],
            "Precision": results["precision_weighted"],
            "Recall": results["recall_weighted"],
            "F1_Weighted": results["f1_weighted"],
            "F1_Macro": results["f1_macro"],
        })

        # Save model
        joblib.dump(trainer.model, PROJECT_ROOT / "models" / "baseline" / f"{name}.pkl")
        print(f"        ✓ Test F1 (weighted): {results['f1_weighted']:.4f}")
        print(f"        ✓ Test F1 (macro):    {results['f1_macro']:.4f}")

        if results["f1_weighted"] > best_f1:
            best_f1 = results["f1_weighted"]
            best_model_name = name

    # Save comparison table
    print("\n[4/5] Saving comparison table...")
    comp_df = pd.DataFrame(results_table)
    comp_df.to_csv(PROJECT_ROOT / "reports" / "tables" / "model_comparison.csv", index=False)
    print("\n" + comp_df.to_string(index=False))

    # Save best model as production
    print("\n[5/5] Saving production model...")
    src = PROJECT_ROOT / "models" / "baseline" / f"{best_model_name}.pkl"
    dst = PROJECT_ROOT / "models" / "production" / "model.pkl"
    shutil.copy(src, dst)
    print(f"      ✓ Best model: {best_model_name} (F1={best_f1:.4f})")
    print(f"      ✓ Saved to: {dst}")

    # Save final artifacts for deployment
    joblib.dump(preprocessor, PROJECT_ROOT / "models" / "production" / "preprocessor.pkl")
    joblib.dump(builder, PROJECT_ROOT / "models" / "production" / "feature_builder.pkl")
    print(f"      ✓ Preprocessor saved to production/")
    print(f"      ✓ FeatureBuilder saved to production/")

    print("\n" + "=" * 70)
    print("✅ TRAINING & EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
