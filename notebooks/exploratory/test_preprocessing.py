#!/usr/bin/env python3
"""Quick test script for the preprocessing pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.preprocessing import TextPreprocessor


def main() -> None:
    """Run preprocessing test."""
    print("=" * 60)
    print("🧪 TESTING: Text Preprocessing Pipeline")
    print("=" * 60)

    # Load sample data
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "tickets.csv")
    sample_texts = df["text"].head(5).tolist()

    print("\n[1/3] Original texts (first 3):")
    for i, text in enumerate(sample_texts[:3], 1):
        print(f"   {i}. {text[:80]}...")

    # Initialize preprocessor
    print("\n[2/3] Initializing TextPreprocessor...")
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatize=True,
    )

    # Transform
    print("\n[3/3] Applying preprocessing...")
    cleaned_texts = preprocessor.transform(sample_texts)

    print("\n📝 Cleaned texts (first 3):")
    for i, text in enumerate(cleaned_texts[:3], 1):
        print(f"   {i}. {text[:80]}...")

    # Save sample
    output_path = PROJECT_ROOT / "data" / "processed" / "sample_cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"original": sample_texts, "cleaned": cleaned_texts}).to_csv(
        output_path, index=False
    )
    print(f"\n✅ Sample saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
