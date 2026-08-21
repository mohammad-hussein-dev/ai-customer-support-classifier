"""Unit tests for model training."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.train_model import ModelTrainer


class TestModelTrainer:
    """Test suite for ModelTrainer."""

    def test_initialization(self):
        trainer = ModelTrainer(model_name="logistic_regression")
        assert trainer.model_name == "logistic_regression"

    def test_unsupported_model(self):
        with pytest.raises(ValueError):
            ModelTrainer(model_name="random_forest")

    def test_cross_validate(self):
        np.random.seed(42)
        X = csr_matrix(np.random.rand(100, 10))
        y = np.random.choice(["A", "B", "C"], size=100)

        trainer = ModelTrainer(model_name="logistic_regression", cv_folds=3)
        results = trainer.cross_validate(X, y)

        assert "test_accuracy" in results
        assert "test_f1_weighted" in results
        assert len(results["test_accuracy"]) == 3

    def test_fit_predict(self):
        np.random.seed(42)
        X = csr_matrix(np.random.rand(50, 10))
        y = np.random.choice(["A", "B"], size=50)

        trainer = ModelTrainer(model_name="naive_bayes")
        trainer.fit(X, y)

        X_test = csr_matrix(np.random.rand(5, 10))
        predictions = trainer.predict(X_test)

        assert len(predictions) == 5
        assert all(isinstance(p, str) for p in predictions)
