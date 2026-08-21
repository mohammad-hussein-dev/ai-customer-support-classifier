#!/usr/bin/env python3
"""Test preprocessing + feature engineering pipeline end-to-end."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessing import TextPreprocessor
from src.features.build_features import FeatureBuilder


def main() -> None:
    """Run end-to-end pipeline test."""
    print("=" * 70)
    print("🔗 TESTING: Preprocessing + Feature Engineering Pipeline")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "tickets.csv")
    print(f"      ✓ Loaded {len(df):,} records")

    # Train/test split (stratified)
    print("\n[2/5] Stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["category"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["category"],
    )
    print(f"      ✓ Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Preprocessing
    print("\n[3/5] Preprocessing texts...")
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatize=True,
    )
    X_train_clean = preprocessor.transform(X_train)
    X_test_clean = preprocessor.transform(X_test)

    print(f"      ✓ Sample original: {X_train[0][:60]}...")
    print(f"      ✓ Sample cleaned:  {X_train_clean[0][:60]}...")

    # Feature Engineering
    print("\n[4/5] Building TF-IDF features...")
    builder = FeatureBuilder(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        include_meta=False,
    )
    X_train_vec = builder.fit_transform(X_train_clean)
    X_test_vec = builder.transform(X_test_clean)

    print(f"      ✓ Vocabulary size: {len(builder.get_feature_names()):,}")
    print(f"      ✓ Train matrix shape: {X_train_vec.shape}")
    print(f"      ✓ Test matrix shape:  {X_test_vec.shape}")
    print(f"      ✓ Matrix density: {X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1]):.4%}")

    # Save artifacts
    print("\n[5/5] Saving artifacts...")
    import joblib

    artifact_dir = PROJECT_ROOT / "models" / "baseline"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, artifact_dir / "preprocessor.pkl")
    joblib.dump(builder, artifact_dir / "feature_builder.pkl")

    # Save labels separately (different lengths)
    pd.DataFrame({"label": y_train}).to_csv(
        PROJECT_ROOT / "data" / "processed" / "y_train.csv", index=False
    )
    pd.DataFrame({"label": y_test}).to_csv(
        PROJECT_ROOT / "data" / "processed" / "y_test.csv", index=False
    )

    print(f"      ✓ Preprocessor saved: {artifact_dir / 'preprocessor.pkl'}")
    print(f"      ✓ FeatureBuilder saved: {artifact_dir / 'feature_builder.pkl'}")
    print(f"      ✓ Labels saved: y_train.csv, y_test.csv")

    print("\n" + "=" * 70)
    print("✅ PIPELINE TEST COMPLETE — Ready for modeling!")
    print("=" * 70)


if __name__ == "__main__":
    main()
