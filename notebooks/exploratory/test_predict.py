#!/usr/bin/env python3
"""Test prediction pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict_model import TicketClassifier


def main() -> None:
    """Run prediction test."""
    print("=" * 60)
    print("🔮 TESTING: Prediction Pipeline")
    print("=" * 60)

    clf = TicketClassifier.from_directory(
        PROJECT_ROOT / "models" / "production",
        review_threshold=0.7,
    )

    test_texts = [
        "I was charged twice for my subscription this month.",
        "The app crashes every time I try to upload a file.",
        "I forgot my password and cannot reset it.",
        "I want a full refund for my purchase made 3 days ago.",
    ]

    print("\nPredictions:")
    for text in test_texts:
        result = clf.predict(text)
        status = "⚠️ REVIEW" if result["needs_review"] else "✅ AUTO"
        print(f"\n  📝 {text[:50]}...")
        print(f"     → Category: {result['category']}")
        print(f"     → Confidence: {result['confidence']:.2%}")
        print(f"     → Status: {status}")

    print("\n" + "=" * 60)
    print("✅ PREDICTION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
