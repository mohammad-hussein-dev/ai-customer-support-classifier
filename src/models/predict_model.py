"""Production inference pipeline for ticket classification."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.data.preprocessing import TextPreprocessor
from src.features.build_features import FeatureBuilder

logger = logging.getLogger(__name__)


class TicketClassifier:
    """Production-ready ticket classification interface."""

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        vectorizer_path: str,
        review_threshold: float = 0.7,
    ) -> None:
        """Initialize classifier with serialized artifacts."""
        logger.info("Loading model artifacts...")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_builder = joblib.load(vectorizer_path)
        self.review_threshold = review_threshold

        if hasattr(self.model, "classes_"):
            self.class_names = list(self.model.classes_)
        else:
            self.class_names = []

        logger.info(
            "Classifier loaded: %d classes, review_threshold=%.2f",
            len(self.class_names), review_threshold,
        )

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify a single support ticket."""
        clean_text = self.preprocessor.clean(text)
        features = self.feature_builder.transform([clean_text])
        prediction = self.model.predict(features)[0]

        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
        elif hasattr(self.model, "decision_function"):
            decisions = self.model.decision_function(features)
            if decisions.ndim == 1:
                probs = 1 / (1 + np.exp(-decisions))
                probabilities = np.array([1 - probs[0], probs[0]])
            else:
                exp_dec = np.exp(decisions - np.max(decisions, keepdims=True))
                probabilities = exp_dec / np.sum(exp_dec, keepdims=True)

        if probabilities is not None:
            confidence = float(np.max(probabilities))
            all_probs = {
                name: float(prob)
                for name, prob in zip(self.class_names, probabilities)
            }
        else:
            confidence = 1.0
            all_probs = {prediction: 1.0}

        needs_review = confidence < self.review_threshold

        return {
            "category": prediction,
            "confidence": round(confidence, 4),
            "needs_review": needs_review,
            "all_probabilities": all_probs,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple tickets efficiently."""
        return [self.predict(t) for t in texts]

    @classmethod
    def from_directory(cls, artifact_dir: str, **kwargs: Any) -> "TicketClassifier":
        """Load classifier from a directory of artifacts."""
        artifact_dir = Path(artifact_dir)
        return cls(
            model_path=str(artifact_dir / "model.pkl"),
            preprocessor_path=str(artifact_dir / "preprocessor.pkl"),
            vectorizer_path=str(artifact_dir / "feature_builder.pkl"),
            **kwargs,
        )
