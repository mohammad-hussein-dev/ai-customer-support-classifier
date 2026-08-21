#!/usr/bin/env python3
"""
Preprocess Hybrid Dataset and Create Stratified Train/Test Split.
Handles class imbalance via stratification.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessing import TextPreprocessor


def load_hybrid_dataset() -> pd.DataFrame:
    """Load the hybrid dataset (real + synthetic)."""
    path = Path("data/raw/hybrid_dataset.csv")
    if not path.exists():
        raise FileNotFoundError(
            "Hybrid dataset not found. Run build_hybrid_dataset.py first."
        )
    
    df = pd.read_csv(path)
    print(f"📥 Loaded hybrid dataset: {len(df)} samples")
    print("   📊 Raw distribution:")
    for cat, count in df["category"].value_counts().items():
        pct = count / len(df) * 100
        print(f"      {cat}: {count} ({pct:.1f}%)")
    return df


def preprocess_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TextPreprocessor to all texts."""
    print("\n🔧 Initializing TextPreprocessor...")
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True
    )
    
    print("🧹 Cleaning texts...")
    df["text_clean"] = df["text"].apply(preprocessor.clean)
    
    # Show before/after example
    print("\n   📝 Example (before → after):")
    print(f"      BEFORE: {df['text'].iloc[0][:80]}...")
    print(f"      AFTER:  {df['text_clean'].iloc[0][:80]}...")
    
    # Remove empty texts after cleaning
    empty_count = (df["text_clean"].str.strip() == "").sum()
    if empty_count > 0:
        print(f"\n   ⚠️  Removed {empty_count} empty texts after cleaning")
        df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)
    
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42
) -> tuple:
    """
    Create stratified train/test split preserving class distribution.
    Critical for imbalanced datasets (Refund only 6.6%).
    
    Args:
        df: Preprocessed dataframe
        test_size: Fraction for test set
        random_state: Reproducibility seed
    
    Returns:
        (train_df, test_df) tuple
    """
    print(f"\n✂️  Stratified Split: {100*(1-test_size):.0f}/{100*test_size:.0f}")
    print("   🎯 Strategy: Preserve class distribution")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["category"],
        shuffle=True
    )
    
    print(f"\n   ✅ Train set: {len(train_df)} samples")
    print("      Distribution:")
    for cat, count in train_df["category"].value_counts().items():
        pct = count / len(train_df) * 100
        print(f"         {cat}: {count} ({pct:.1f}%)")
    
    print(f"\n   ✅ Test set: {len(test_df)} samples")
    print("      Distribution:")
    for cat, count in test_df["category"].value_counts().items():
        pct = count / len(test_df) * 100
        print(f"         {cat}: {count} ({pct:.1f}%)")
    
    return train_df, test_df


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save processed splits to disk."""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    # Save as numpy for model training (faster loading)
    np.save(output_dir / "y_train.npy", train_df["category"].values)
    np.save(output_dir / "y_test.npy", test_df["category"].values)
    
    print(f"\n💾 Saved to {output_dir}/")
    print("   📄 train.csv / test.csv")
    print("   🔢 y_train.npy / y_test.npy")


def main() -> None:
    """Main pipeline: load → preprocess → split → save."""
    print("=" * 60)
    print("🔬 Preprocess and Split Hybrid Dataset")
    print("=" * 60)
    
    # 1. Load
    df = load_hybrid_dataset()
    
    # 2. Preprocess
    df = preprocess_texts(df)
    
    # 3. Split
    train_df, test_df = stratified_split(df, test_size=0.3)
    
    # 4. Save
    save_splits(train_df, test_df)
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing Complete!")
    print(f"   📊 Total samples: {len(df)}")
    print(f"   🚂 Train: {len(train_df)}")
    print(f"   🧪 Test:  {len(test_df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
