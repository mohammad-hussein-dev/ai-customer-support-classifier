"""Comprehensive model evaluation and reporting.

Generates classification reports, confusion matrices, and performance
visualizations for ticket classification models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for customer support ticket classifiers.

    Produces detailed evaluation metrics, confusion matrices, and
    confidence analysis for production readiness assessment.

    Attributes:
        class_names: Ordered list of category names.
        output_dir: Directory to save evaluation artifacts.
    """

    def __init__(
        self,
        class_names: List[str],
        output_dir: str = "reports/figures",
    ) -> None:
        """Initialize evaluator.

        Args:
            class_names: List of class labels in order.
            output_dir: Path to save figures and reports.
        """
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run complete evaluation pipeline.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).

        Returns:
            Dictionary containing all evaluation results.
        """
        logger.info("Starting model evaluation")

        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        results = {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "f1_macro": macro_f1,
            "classification_report": classification_report(
                y_true, y_pred, target_names=self.class_names, zero_division=0
            ),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        results["confusion_matrix"] = cm
        self._plot_confusion_matrix(cm)

        # Confidence analysis
        if y_proba is not None:
            max_conf = np.max(y_proba, axis=1)
            results["mean_confidence"] = float(np.mean(max_conf))
            results["median_confidence"] = float(np.median(max_conf))
            self._plot_confidence_distribution(y_proba)

        # Error analysis
        errors = self._analyze_errors(y_true, y_pred)
        results["error_patterns"] = errors

        # Save report
        self._save_report(results)

        logger.info("Evaluation complete. Artifacts saved to %s", self.output_dir)
        return results

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Plot and save normalized confusion matrix."""
        plt.figure(figsize=(10, 8))

        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Proportion"},
            linewidths=0.5,
        )

        plt.title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()

        save_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("Confusion matrix saved")

    def _plot_confidence_distribution(self, y_proba: np.ndarray) -> None:
        """Plot prediction confidence distribution."""
        max_conf = np.max(y_proba, axis=1)

        plt.figure(figsize=(10, 6))
        sns.histplot(max_conf, bins=20, kde=True, color="steelblue")

        plt.axvline(x=0.7, color="red", linestyle="--", label="Review Threshold (0.7)")
        plt.axvline(x=np.mean(max_conf), color="green", linestyle="-", label=f"Mean ({np.mean(max_conf):.3f})")

        plt.title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Maximum Predicted Probability", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend()
        plt.tight_layout()

        save_path = self.output_dir / "confidence_distribution.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("Confidence distribution saved")

    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, List[int]]:
        """Identify and categorize misclassification patterns."""
        errors = {}
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if true != pred:
                key = f"{true} -> {pred}"
                errors.setdefault(key, []).append(i)
        return errors

    def _save_report(self, results: Dict[str, Any]) -> None:
        """Save evaluation report to text file."""
        report_path = self.output_dir.parent / "tables" / "evaluation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Accuracy:           {results['accuracy']:.4f}\n")
            f.write(f"Precision (weighted): {results['precision_weighted']:.4f}\n")
            f.write(f"Recall (weighted):    {results['recall_weighted']:.4f}\n")
            f.write(f"F1 (weighted):        {results['f1_weighted']:.4f}\n")
            f.write(f"F1 (macro):           {results['f1_macro']:.4f}\n\n")
            if "mean_confidence" in results:
                f.write(f"Mean Confidence:      {results['mean_confidence']:.4f}\n")
                f.write(f"Median Confidence:    {results['median_confidence']:.4f}\n\n")
            f.write("-" * 60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("-" * 60 + "\n")
            f.write(results["classification_report"])
            f.write("\n")

        logger.info("Report saved to %s", report_path)
