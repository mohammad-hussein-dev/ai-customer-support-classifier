#!/usr/bin/env python3
"""
Build Hybrid Dataset: Banking77 (real, from GitHub) + Synthetic
Maps 77 banking intents → 4 categories: Account, Billing, Technical, Refund
"""

import pandas as pd
import json
import random
import os
import urllib.request
from pathlib import Path

random.seed(42)

# ═══════════════════════════════════════════════════════════
# Mapping 77 Banking77 intents → 4 categories
# ═══════════════════════════════════════════════════════════
INTENT_MAP = {
    # ─── Account (حساب کاربری) ───
    "activate_my_card": "Account",
    "age_limit": "Account",
    "card_about_to_expire": "Account",
    "card_acceptance": "Account",
    "card_arrival": "Account",
    "card_delivery_estimate": "Account",
    "card_linking": "Account",
    "contactless_not_working": "Account",
    "country_support": "Account",
    "declined_card_payment": "Account",
    "declined_cash_withdrawal": "Account",
    "disposable_card_limits": "Account",
    "edit_personal_details": "Account",
    "exchange_charge": "Account",
    "exchange_rate": "Account",
    "getting_spare_card": "Account",
    "getting_virtual_card": "Account",
    "lost_or_stolen_card": "Account",
    "order_physical_card": "Account",
    "passcode_forgotten": "Account",
    "pending_card_payment": "Account",
    "pending_cash_withdrawal": "Account",
    "pending_top_up": "Account",
    "pin_blocked": "Account",
    "receiving_money": "Account",
    "supported_cards_and_currencies": "Account",
    "top_up_limits": "Account",
    "top_up_reverted": "Account",
    "unable_to_verify_identity": "Account",
    "verify_my_identity": "Account",
    "virtual_card_not_working": "Account",
    "why_verify_identity": "Account",
    "wrong_amount_of_cash_received": "Account",
    
    # ─── Billing (پرداخت و صورتحساب) ───
    "balance_not_updated_after_bank_transfer": "Billing",
    "balance_not_updated_after_cheque_or_cash_deposit": "Billing",
    "bank_transfer_charge": "Billing",
    "card_payment_fee_charged": "Billing",
    "card_payment_wrong_amount": "Billing",
    "card_payment_not_recognised": "Billing",
    "card_swallowed": "Billing",
    "cash_withdrawal_charge": "Billing",
    "cash_withdrawal_not_recognized": "Billing",
    "change_pin": "Billing",
    "compromised_card": "Billing",
    "declined_transfer": "Billing",
    "direct_debit_payment_not_recognised": "Billing",
    "extra_charge_on_statement": "Billing",
    "failed_transfer": "Billing",
    "fiat_currency_support": "Billing",
    "pending_transfer": "Billing",
    "reverted_bank_transfer": "Billing",
    "top_up_by_bank_transfer_charge": "Billing",
    "top_up_by_card_charge": "Billing",
    "top_up_by_debit_card": "Billing",
    "transaction_charged_twice": "Billing",
    "transfer_fee_charged": "Billing",
    "transfer_into_account": "Billing",
    "transfer_not_received_by_recipient": "Billing",
    "transfer_timing": "Billing",
    "wrong_exchange_rate_for_cash_withdrawal": "Billing",
    "wrong_recipient": "Billing",
    
    # ─── Technical Support (مشکلات فنی) ───
    "apple_pay_or_google_pay": "Technical Support",
    "atm_support": "Technical Support",
    "automatic_top_up": "Technical Support",
    "beneficiary_not_allowed": "Technical Support",
    "cancel_transfer": "Technical Support",
    "exchange_via_app": "Technical Support",
    "top_up_by_cash_or_cheque": "Technical Support",
    "top_up_failed": "Technical Support",
    "verify_top_up": "Technical Support",
    "visa_or_mastercard": "Technical Support",
    
    # ─── Refund (استرداد) ───
    "request_refund": "Refund",
    "reverted_card_payment": "Refund",
}


def add_realistic_noise(text: str, noise_level: float = 0.15) -> str:
    """
    Add realistic noise to simulate real customer tickets.
    Includes: typos, abbreviations, extra punctuation, casing issues.
    
    Args:
        text: Clean text from dataset
        noise_level: Probability of applying noise (0-1)
    
    Returns:
        Noisy text string
    """
    if random.random() > noise_level:
        return text
    
    typos = {
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
    
    words = text.split()
    noisy_words = []
    
    for word in words:
        w_lower = word.lower()
        if w_lower in typos and random.random() < 0.3:
            noisy_words.append(random.choice(typos[w_lower]))
        else:
            noisy_words.append(word)
    
    abbrevs = {
        "as soon as possible": "asap",
        "by the way": "btw",
        "for your information": "fyi",
    }
    
    result = " ".join(noisy_words)
    for full, abbr in abbrevs.items():
        if full in result.lower() and random.random() < 0.2:
            result = result.replace(full, abbr)
    
    if random.random() < 0.2:
        result = result.replace("!", "!!").replace("?", "???")
    if random.random() < 0.15:
        result = result.upper()
    elif random.random() < 0.15:
        result = result.lower()
    
    return result


def download_banking77() -> pd.DataFrame:
    """
    Download Banking77 dataset directly from GitHub.
    No external libraries needed — uses urllib (built-in).
    """
    print("📥 Downloading Banking77 from GitHub...")
    
    # URLs for raw data
    base_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
    train_url = f"{base_url}/train.csv"
    test_url = f"{base_url}/test.csv"
    categories_url = f"{base_url}/categories.json"
    
    cache_dir = Path(".cache/banking77")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download files
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
    
    # Load categories mapping
    with open(cache_dir / "categories.json", "r", encoding="utf-8") as f:
        categories = json.load(f)
    
    # Load train + test
    train_df = pd.read_csv(cache_dir / "train.csv")
    test_df = pd.read_csv(cache_dir / "test.csv")
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    # Map categories
    combined["category"] = combined["category"].apply(
        lambda x: INTENT_MAP.get(x, "Account")
    )
    
    # Rename and select
    combined = combined.rename(columns={"text": "text"})
    combined = combined[["text", "category"]]
    combined["source"] = "banking77_real"
    combined["ticket_id"] = [f"B77_{i:05d}" for i in range(len(combined))]
    
    print(f"   ✅ Loaded {len(combined)} real samples")
    print("   📊 Distribution:")
    for cat, count in combined["category"].value_counts().items():
        print(f"      {cat}: {count}")
    
    return combined


def load_synthetic() -> pd.DataFrame:
    """Load existing synthetic dataset (Refund only)."""
    synthetic_path = Path("data/raw/tickets.csv")
    if not synthetic_path.exists():
        print("   ⚠️  Synthetic data not found, skipping...")
        return pd.DataFrame()
    
    df = pd.read_csv(synthetic_path)
    if "text" not in df.columns:
        df = df.rename(columns={"text": "text"})
    df = df[["text", "category"]]
    df["source"] = "synthetic"
    df["ticket_id"] = [f"SYN_{i:05d}" for i in range(len(df))]
    df = df[df["category"] == "Refund"]
    
    print(f"   ✅ Loaded {len(df)} synthetic Refund samples")
    return df


def build_hybrid() -> pd.DataFrame:
    """Combine real + synthetic data with noise injection."""
    print("=" * 60)
    print("🔬 Building Hybrid Dataset")
    print("=" * 60)
    
    real_df = download_banking77()
    synth_df = load_synthetic()
    
    print("\n🎲 Adding realistic noise (15% probability)...")
    real_df["text"] = real_df["text"].apply(
        lambda x: add_realistic_noise(x, noise_level=0.15)
    )
    
    hybrid = pd.concat([real_df, synth_df], ignore_index=True)
    hybrid["ticket_id"] = [f"TICKET_{i:06d}" for i in range(len(hybrid))]
    hybrid = hybrid.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs("data/raw", exist_ok=True)
    hybrid.to_csv("data/raw/hybrid_dataset.csv", index=False)
    
    print("\n" + "=" * 60)
    print("✅ Hybrid Dataset Saved!")
    print(f"   📁 Path: data/raw/hybrid_dataset.csv")
    print(f"   📊 Total: {len(hybrid)} samples")
    print("\n   Category Distribution:")
    for cat, count in hybrid["category"].value_counts().items():
        pct = count / len(hybrid) * 100
        print(f"      {cat}: {count} ({pct:.1f}%)")
    print("=" * 60)
    
    return hybrid


if __name__ == "__main__":
    build_hybrid()
