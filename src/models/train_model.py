"""Model training pipeline with cross-validation and hyperparameter tuning.

Supports multiple classifiers with stratified cross-validation for
imbalanced datasets.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer for text classification models.

    Handles model initialization, cross-validation, and final fitting
    with best hyperparameters.

    Attributes:
        model_name: Name of the classifier to use.
        model: The underlying sklearn estimator.
        cv_results: Cross-validation results from last training.
    """

    SUPPORTED_MODELS = {
        "logistic_regression": LogisticRegression,
        "naive_bayes": MultinomialNB,
        "svm": LinearSVC,
    }

    def __init__(
        self,
        model_name: str = "logistic_regression",
        model_params: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        """Initialize the model trainer.

        Args:
            model_name: Classifier name. Must be in SUPPORTED_MODELS.
            model_params: Hyperparameters for the classifier.
            cv_folds: Number of cross-validation folds.
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If model_name is not supported.
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_params = model_params or {}
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv_results: Optional[Dict[str, Any]] = None

        # Set random state if applicable
        model_cls = self.SUPPORTED_MODELS[model_name]
        if "random_state" in model_cls.__init__.__code__.co_varnames:
            self.model_params.setdefault("random_state", random_state)

        self.model = model_cls(**self.model_params)
        logger.info(
            "ModelTrainer: %s with params %s",
            model_name,
            self.model_params,
        )

    def cross_validate(
        self,
        X: csr_matrix,
        y: np.ndarray,
        scoring: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform stratified cross-validation.

        Args:
            X: Feature matrix.
            y: Target vector.
            scoring: List of scoring metrics.

        Returns:
            Dictionary with cross-validation scores.
        """
        if scoring is None:
            scoring = [
                "accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
            ]

        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        logger.info("Starting %d-fold stratified cross-validation", self.cv_folds)

        self.cv_results = cross_validate(
            self.model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        # Log mean scores
        for metric in scoring:
            mean_score = np.mean(self.cv_results[f"test_{metric}"])
            std_score = np.std(self.cv_results[f"test_{metric}"])
            logger.info(
                "CV %s: %.4f (+/- %.4f)",
                metric,
                mean_score,
                std_score,
            )

        return self.cv_results

    def fit(self, X: csr_matrix, y: np.ndarray) -> "ModelTrainer":
        """Fit the model on the full training set.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting %s on %d samples", self.model_name, X.shape[0])
        self.model.fit(X, y)
        logger.info("Model fitted successfully")
        return self

    def predict(self, X: csr_matrix) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: csr_matrix) -> Optional[np.ndarray]:
        """Generate probability estimates if supported.

        Args:
            X: Feature matrix.

        Returns:
            Probability array or None if model doesn't support probabilities.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            decisions = self.model.decision_function(X)
            if decisions.ndim == 1:
                probs = 1 / (1 + np.exp(-decisions))
                return np.column_stack([1 - probs, probs])
            else:
                exp_decisions = np.exp(
                    decisions - np.max(decisions, axis=1, keepdims=True)
                )
                return exp_decisions / np.sum(exp_decisions, axis=1, keepdims=True)
        else:
            logger.warning("Model does not support probability estimates")
            return None
