#!/usr/bin/env python3
"""Exploratory Data Analysis (EDA) script for customer support tickets.

This script performs comprehensive EDA on the synthetic ticket dataset
and saves all visualizations to reports/figures/.

Usage:
    $ python notebooks/exploratory/run_eda.py
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.visualization.visualize import TicketVisualizer


def main() -> None:
    """Run the complete EDA pipeline."""
    print("=" * 70)
    print("🔬 EXPLORATORY DATA ANALYSIS - Customer Support Tickets")
    print("=" * 70)

    # ============================================================
    # 1. Load Data
    # ============================================================
    print("\n[1/8] Loading dataset...")
    data_path = PROJECT_ROOT / "data" / "raw" / "tickets.csv"
    df = pd.read_csv(data_path)
    print(f"      ✓ Loaded {len(df):,} records from {data_path}")

    # ============================================================
    # 2. Basic Overview
    # ============================================================
    print("\n[2/8] Dataset Overview:")
    print(f"      Shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")
    print(f"      Missing values:\n{df.isnull().sum().to_string().replace(chr(10), chr(10)+' '*6)}")
    print(f"      Duplicated rows: {df.duplicated().sum()}")
    print(f"\n      Data types:")
    for col, dtype in df.dtypes.items():
        print(f"      • {col:15s}: {dtype}")

    # ============================================================
    # 3. Summary Statistics
    # ============================================================
    print("\n[3/8] Text Statistics:")
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    print(f"      Avg characters: {df['char_count'].mean():.1f}")
    print(f"      Avg words: {df['word_count'].mean():.1f}")
    print(f"      Min words: {df['word_count'].min()}")
    print(f"      Max words: {df['word_count'].max()}")

    # Save summary to CSV
    summary = df.groupby("category").agg({
        "char_count": ["mean", "std", "min", "max"],
        "word_count": ["mean", "std", "min", "max"],
        "ticket_id": "count",
    }).round(2)
    summary_path = PROJECT_ROOT / "reports" / "tables" / "eda_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path)
    print(f"      ✓ Summary saved to {summary_path}")

    # ============================================================
    # 4. Initialize Visualizer
    # ============================================================
    print("\n[4/8] Initializing visualizer...")
    viz = TicketVisualizer(output_dir=str(PROJECT_ROOT / "reports" / "figures"))

    # ============================================================
    # 5. Generate Visualizations
    # ============================================================
    print("\n[5/8] Generating visualizations...")

    print("      → Category distribution (bar)...")
    viz.plot_category_distribution(df)

    print("      → Category distribution (pie)...")
    viz.plot_category_pie(df)

    print("      → Priority distribution...")
    viz.plot_priority_distribution(df)

    print("      → Priority vs Category heatmap...")
    viz.plot_priority_by_category(df)

    print("      → Text length analysis...")
    viz.plot_text_length_distribution(df)

    print("      → Word clouds...")
    for category in df["category"].unique():
        texts = df[df["category"] == category]["text"].tolist()
        viz.plot_wordcloud(texts, category)

    print("      → Temporal trends...")
    viz.plot_temporal_trends(df)

    print("      → Class imbalance analysis...")
    viz.plot_class_imbalance(df)

    print("      → N-gram analysis...")
    for category in df["category"].unique():
        viz.plot_ngrams(df, category, n=1, top_k=15)
        viz.plot_ngrams(df, category, n=2, top_k=15)

    # ============================================================
    # 6. Final Report
    # ============================================================
    print("\n" + "=" * 70)
    print("✅ EDA COMPLETE")
    print("=" * 70)
    print(f"\n📊 Figures saved to: reports/figures/")
    print(f"📋 Summary saved to: reports/tables/eda_summary.csv")
    print(f"\n📁 Generated files:")
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    for f in sorted(figures_dir.glob("*.png")):
        print(f"   • {f.name}")


if __name__ == "__main__":
    main()
