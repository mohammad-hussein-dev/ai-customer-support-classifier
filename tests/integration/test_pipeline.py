"""Integration tests for end-to-end pipeline."""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessing import TextPreprocessor
from src.features.build_features import FeatureBuilder
from src.models.predict_model import TicketClassifier
from src.models.train_model import ModelTrainer


class TestPipeline:
    """End-to-end pipeline integration tests."""

    def test_full_pipeline(self, tmp_path):
        df = pd.DataFrame(
            {
                "text": [
                    "I was charged twice for my subscription",
                    "The app crashes when I upload files",
                    "I forgot my password and cannot login",
                    "I want a refund for my purchase",
                ]
                * 5,
                "category": ["Billing", "Technical Support", "Account", "Refund"] * 5,
            }
        )

        preprocessor = TextPreprocessor()
        texts_clean = preprocessor.transform(df["text"].tolist())

        builder = FeatureBuilder(max_features=50)
        X = builder.fit_transform(texts_clean)
        y = np.array(df["category"].tolist())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        trainer = ModelTrainer(model_name="logistic_regression")
        trainer.fit(X_train, y_train)

        y_pred = trainer.predict(X_test)
        assert len(y_pred) == len(y_test)

        model_path = tmp_path / "model.pkl"
        prep_path = tmp_path / "preprocessor.pkl"
        feat_path = tmp_path / "feature_builder.pkl"

        joblib.dump(trainer.model, model_path)
        joblib.dump(preprocessor, prep_path)
        joblib.dump(builder, feat_path)

        clf = TicketClassifier(
            model_path=str(model_path),
            preprocessor_path=str(prep_path),
            vectorizer_path=str(feat_path),
        )
        result = clf.predict("I was charged twice")
        assert "category" in result
        assert "confidence" in result
        assert "needs_review" in result
