#!/usr/bin/env python3
"""Build a hybrid dataset with exactly 8 specific Banking77 intents.

This script downloads the Banking77 dataset, filters only the 8 intents
required for the project, and saves the result as a CSV file.
No mapping to high-level categories is performed.
"""

import json
import os
import random
import urllib.request
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# ============================================================================
# CONSTANTS (Google Style: UPPER_CASE for module-level constants)
# ============================================================================

# The 8 intents required by the project specification.
SELECTED_INTENTS: List[str] = [
    "card_arrival",
    "card_not_working",
    "cash_withdrawal_not_recognised",
    "declined_card_payment",
    "lost_or_stolen_card",
    "transaction_charged_twice",
    "transfer_not_received_by_recipient",
    "cash_withdrawal_charge",
]

RANDOM_SEED: int = 42
NOISE_PROBABILITY: float = 0.15  # 15% chance to apply noise to each ticket

# ----------------------------------------------------------------------------
# Noise dictionaries (typos, abbreviations, etc.)
# ----------------------------------------------------------------------------

_TYPOS: dict = {
    "account": ["acount", "accunt", "accout"],
    "payment": ["paymnt", "paymet", "pament"],
    "transfer": ["transfr", "transer", "trnsfer"],
    "card": ["crd", "cad", "crad"],
    "charged": ["chaged", "chrged", "chargd"],
    "refund": ["refnd", "refund"],
    "please": ["pls", "plz", "plese"],
    "thanks": ["thx", "tks", "thanx"],
    "hello": ["hi", "hey", "hii"],
    "help": ["hlp", "hel", "hep"],
}

_ABBREVIATIONS: dict = {
    "as soon as possible": "asap",
    "by the way": "btw",
    "for your information": "fyi",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _add_realistic_noise(text: str, noise_level: float = NOISE_PROBABILITY) -> str:
    """Add realistic noise to a clean text.

    Args:
        text: Clean text from the dataset.
        noise_level: Probability (0-1) of applying noise.

    Returns:
        Noisy version of the input text.
    """
    if random.random() > noise_level:
        return text

    words = text.split()
    noisy_words: List[str] = []

    for word in words:
        w_lower = word.lower()
        if w_lower in _TYPOS and random.random() < 0.3:
            noisy_words.append(random.choice(_TYPOS[w_lower]))
        else:
            noisy_words.append(word)

    result = " ".join(noisy_words)

    # Replace full phrases with abbreviations
    for full, abbr in _ABBREVIATIONS.items():
        if full in result.lower() and random.random() < 0.2:
            result = result.replace(full, abbr)

    # Extra punctuation and case changes
    if random.random() < 0.2:
        result = result.replace("!", "!!").replace("?", "???")
    if random.random() < 0.15:
        result = result.upper()
    elif random.random() < 0.15:
        result = result.lower()

    return result


# ============================================================================
# MAIN DATA LOADING FUNCTIONS
# ============================================================================

def download_banking77() -> pd.DataFrame:
    """Download Banking77 dataset from GitHub and filter the 8 intents.

    Returns:
        DataFrame with columns: text, category (intent name), source, ticket_id.
    """
    print("📥 Downloading Banking77 from GitHub...")

    base_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
    train_url = f"{base_url}/train.csv"
    test_url = f"{base_url}/test.csv"
    categories_url = f"{base_url}/categories.json"

    cache_dir = Path(".cache/banking77")
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "train": (train_url, cache_dir / "train.csv"),
        "test": (test_url, cache_dir / "test.csv"),
        "categories": (categories_url, cache_dir / "categories.json"),
    }

    for name, (url, path) in files.items():
        if not path.exists():
            print(f"   ⬇️  Downloading {name}.csv...")
            urllib.request.urlretrieve(url, path)
            print(f"      ✅ Saved to {path}")
        else:
            print(f"   📦 Using cached {name}.csv")

    # Load train + test
    train_df = pd.read_csv(cache_dir / "train.csv")
    test_df = pd.read_csv(cache_dir / "test.csv")
    combined = pd.concat([train_df, test_df], ignore_index=True)

    # 🔥 KEY CHANGE: Keep only the 8 selected intents.
    # The column 'category' in the raw dataset is the intent name.
    combined = combined[combined["category"].isin(SELECTED_INTENTS)]

    # Rename columns for consistency
    combined = combined.rename(columns={"text": "text"})
    combined["source"] = "banking77_real"
    combined["ticket_id"] = [f"B77_{i:05d}" for i in range(len(combined))]

    print(f"   ✅ Loaded {len(combined)} real samples (filtered to 8 intents)")
    print("   📊 Distribution:")
    for intent, count in combined["category"].value_counts().items():
        print(f"      {intent}: {count}")

    return combined


def load_synthetic() -> pd.DataFrame:
    """Load synthetic data (if any) and filter only if it matches the 8 intents.

    Returns:
        DataFrame with synthetic samples, or empty DataFrame if none found.
    """
    synthetic_path = Path("data/raw/tickets.csv")
    if not synthetic_path.exists():
        print("   ⚠️  No synthetic data found. Skipping.")
        return pd.DataFrame()

    df = pd.read_csv(synthetic_path)
    if "text" not in df.columns:
        df = df.rename(columns={"text": "text"})

    # If the synthetic data has a 'category' column, keep only those that are
    # in the 8 selected intents (likely "Refund" is not, so it will be dropped).
    if "category" in df.columns:
        df = df[df["category"].isin(SELECTED_INTENTS)]

    df["source"] = "synthetic"
    df["ticket_id"] = [f"SYN_{i:05d}" for i in range(len(df))]

    print(f"   ✅ Loaded {len(df)} synthetic samples matching the 8 intents")
    return df


def build_hybrid() -> pd.DataFrame:
    """Combine real + synthetic data, add noise, and save.

    Returns:
        The final hybrid DataFrame.
    """
    print("=" * 60)
    print("🔬 Building Hybrid Dataset (8 Banking77 Intents)")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    real_df = download_banking77()
    synth_df = load_synthetic()

    # Add noise to real data (but not synthetic, as it is already noisy)
    print(f"\n🎲 Adding realistic noise ({NOISE_PROBABILITY*100:.0f}% probability)...")
    real_df["text"] = real_df["text"].apply(
        lambda x: _add_realistic_noise(x, noise_level=NOISE_PROBABILITY)
    )

    # Combine
    hybrid = pd.concat([real_df, synth_df], ignore_index=True)
    hybrid["ticket_id"] = [f"TICKET_{i:06d}" for i in range(len(hybrid))]
    hybrid = hybrid.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Save
    os.makedirs("data/raw", exist_ok=True)
    hybrid.to_csv("data/raw/hybrid_dataset.csv", index=False)

    print("\n" + "=" * 60)
    print("✅ Hybrid Dataset Saved!")
    print(f"   📁 Path: data/raw/hybrid_dataset.csv")
    print(f"   📊 Total: {len(hybrid)} samples")
    print("\n   Intent Distribution (exactly 8 intents):")
    for intent, count in hybrid["category"].value_counts().items():
        pct = count / len(hybrid) * 100
        print(f"      {intent}: {count} ({pct:.1f}%)")
    print("=" * 60)

    return hybrid


if __name__ == "__main__":
    build_hybrid()
