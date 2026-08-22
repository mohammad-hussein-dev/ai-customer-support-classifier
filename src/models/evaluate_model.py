"""
Professional Model Evaluation with Rich Terminal Output
=======================================================
Dark-theme figures + beautiful console tables via Rich.

Author: Mohammad Hussein
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.visualization.visualize import BankingVisualizer

logger = logging.getLogger(__name__)
console = Console()


class ModelEvaluator:
    """Professional evaluator for banking intent classifiers.
    
    Produces:
    - Rich terminal reports with color-coded metrics
    - Dark-theme evaluation figures
    - Structured JSON reports
    - Error analysis with business context
    """

    def __init__(
        self,
        class_names: List[str],
        output_dir: str = "reports",
        business_critical: Optional[List[str]] = None,
    ) -> None:
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.business_critical = business_critical or []
        self.visualizer = BankingVisualizer(output_dir=self.figures_dir)
        
        logger.info("ModelEvaluator initialized | classes=%d", len(class_names))

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        texts: Optional[List[str]] = None,
        model=None,
        vectorizer=None,
    ) -> Dict[str, Any]:
        """Run complete evaluation pipeline with rich output."""
        
        console.print()
        console.rule("[bold cyan]🤖 BANKING INTENT CLASSIFIER — EVALUATION REPORT", style="cyan")
        console.print()

        # ── Overall Metrics ──
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        # ── Rich Summary Panel ──
        summary = Table.grid(expand=True)
        summary.add_column(style="bold cyan", justify="right")
        summary.add_column(style="white")
        summary.add_row("📊 Accuracy:", f"{accuracy:.4f}")
        summary.add_row("🎯 Macro-F1:", f"{macro_f1:.4f}")
        summary.add_row("⚖️ Weighted-F1:", f"{weighted_f1:.4f}")
        summary.add_row("📁 Test Samples:", f"{len(y_true):,}")
        summary.add_row("🏷️ Classes:", f"{len(self.class_names)}")
        
        console.print(Panel(summary, title="[bold green]Overall Performance", border_style="green", width=50))
        console.print()

        # ── Per-Class Metrics Table ──
        self._print_class_metrics(y_true, y_pred)
        
        # ── Confusion Matrix + Evaluation Dashboard ──
        self.visualizer.plot_evaluation_dashboard(y_true, y_pred, self.class_names, y_proba)
        
        # ── TF-IDF Terms (if model provided) ──
        if model is not None and vectorizer is not None and hasattr(model, "coef_"):
            self.visualizer.plot_tfidf_terms(vectorizer, model.coef_, self.class_names)
        
        # ── Confidence Analysis ──
        if y_proba is not None:
            self.visualizer.plot_confidence_analysis(y_true, y_pred, y_proba, self.class_names)
        
        # ── Error Analysis ──
        if texts is not None and y_proba is not None:
            self.visualizer.plot_error_patterns(y_true, y_pred, texts, self.class_names, y_proba)
            self._print_error_analysis(y_true, y_pred, texts, y_proba)
        
        # ── Business-Critical Analysis ──
        if self.business_critical:
            self._print_business_analysis(y_true, y_pred)
        
        # ── Save JSON Report ──
        results = self._build_results_dict(y_true, y_pred, y_proba)
        self._save_json_report(results)
        
        console.rule("[bold green]✅ Evaluation Complete", style="green")
        console.print(f"[dim]Artifacts saved to: {self.output_dir}[/dim]")
        console.print()
        
        return results

    def _print_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Print beautiful per-class metrics table."""
        table = Table(
            title="[bold cyan]Per-Class Performance Metrics",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
            row_styles=["none", "dim"],
        )
        table.add_column("Intent", style="bold", min_width=20)
        table.add_column("Precision", justify="center", min_width=10)
        table.add_column("Recall", justify="center", min_width=10)
        table.add_column("F1-Score", justify="center", min_width=10)
        table.add_column("Support", justify="center", min_width=8)
        table.add_column("Status", justify="center", min_width=12)

        precisions = precision_score(y_true, y_pred, average=None, zero_division=0, labels=self.class_names)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0, labels=self.class_names)
        f1s = f1_score(y_true, y_pred, average=None, zero_division=0, labels=self.class_names)
        
        for cls, p, r, f1 in zip(self.class_names, precisions, recalls, f1s):
            support = int(np.sum(y_true == cls))
            is_critical = cls in self.business_critical
            
            # Color-code F1
            if f1 >= 0.90:
                f1_style = "[bold green]"
                status = "🟢 Excellent"
            elif f1 >= 0.75:
                f1_style = "[bold yellow]"
                status = "🟡 Good"
            else:
                f1_style = "[bold red]"
                status = "🔴 Needs Work"
            
            name = cls.replace("_", " ").title()
            if is_critical:
                name = f"[bold red]⚠ {name}[/bold red]"
            
            table.add_row(
                name,
                f"{p:.4f}",
                f"{r:.4f}",
                f"{f1_style}{f1:.4f}[/]",
                str(support),
                status,
            )
        
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        table.add_row("[bold]Macro Average[/bold]", "", "", f"[bold cyan]{macro_f1:.4f}[/bold cyan]", "", "", style="bold")
        
        console.print(table)
        console.print()

    def _print_error_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        texts: List[str],
        y_proba: np.ndarray,
    ) -> None:
        """Print error analysis with example texts."""
        console.rule("[bold yellow]🔍 ERROR ANALYSIS", style="yellow")
        console.print()
        
        errors = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            if t != p:
                errors.append({
                    "index": i,
                    "true": t,
                    "pred": p,
                    "text": texts[i],
                    "confidence": float(np.max(y_proba[i])),
                })
        
        # Group by confusion pair
        from collections import defaultdict
        pairs = defaultdict(list)
        for e in errors:
            pairs[(e["true"], e["pred"])].append(e)
        
        # Sort by frequency
        sorted_pairs = sorted(pairs.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        
        for (true_lbl, pred_lbl), errs in sorted_pairs:
            avg_conf = np.mean([e["confidence"] for e in errs])
            
            panel_content = Table.grid(expand=True)
            panel_content.add_column(style="dim", width=8)
            panel_content.add_column(style="white")
            
            for e in errs[:3]:
                txt = e["text"][:100] + "..." if len(e["text"]) > 100 else e["text"]
                panel_content.add_row(f"[{e['confidence']:.3f}]", txt)
            
            if len(errs) > 3:
                panel_content.add_row("", f"[dim]... and {len(errs)-3} more[/dim]")
            
            title = f"[bold red]{true_lbl.replace('_', ' ').title()} → {pred_lbl.replace('_', ' ').title()}[/bold red]"
            subtitle = f"[dim]{len(errs)} cases | Avg confidence: {avg_conf:.3f}[/dim]"
            
            console.print(Panel(panel_content, title=f"{title}\n{subtitle}", border_style="red", width=100))
        
        console.print()

    def _print_business_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Analyze business-critical intents."""
        console.rule("[bold red]⚠️ BUSINESS-CRITICAL INTENT ANALYSIS", style="red")
        console.print()
        
        for intent in self.business_critical:
            mask = y_true == intent
            if not np.any(mask):
                continue
            
            true_pos = np.sum((y_true == intent) & (y_pred == intent))
            false_neg = np.sum((y_true == intent) & (y_pred != intent))
            false_pos = np.sum((y_true != intent) & (y_pred == intent))
            
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            
            table = Table(title=f"[bold red]{intent.replace('_', ' ').title()}[/bold red]", box=box.SIMPLE, border_style="red")
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="center")
            table.add_column("Risk Level", justify="center")
            
            risk_fn = "🔴 HIGH" if false_neg > 0 else "🟢 LOW"
            risk_fp = "🟡 MED" if false_pos > 0 else "🟢 LOW"
            
            table.add_row("False Negatives (Missed)", str(false_neg), risk_fn)
            table.add_row("False Positives (Misrouted)", str(false_pos), risk_fp)
            table.add_row("Recall (Catch Rate)", f"{recall:.4f}", "")
            table.add_row("Precision (Accuracy)", f"{precision:.4f}", "")
            
            console.print(table)
            console.print("[dim]False negatives = tickets that SHOULD have been routed here but weren't.[/dim]")
            console.print("[dim]These are the most costly errors for business-critical intents.[/dim]")
            console.print()

    def _build_results_dict(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Build structured results dictionary."""
        precisions = precision_score(y_true, y_pred, average=None, zero_division=0, labels=self.class_names)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0, labels=self.class_names)
        f1s = f1_score(y_true, y_pred, average=None, zero_division=0, labels=self.class_names)
        
        per_class = {}
        for cls, p, r, f1 in zip(self.class_names, precisions, recalls, f1s):
            per_class[cls] = {
                "precision": float(p),
                "recall": float(r),
                "f1_score": float(f1),
                "support": int(np.sum(y_true == cls)),
            }
        
        results = {
            "overall": {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            },
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=self.class_names).tolist(),
        }
        
        if y_proba is not None:
            max_conf = np.max(y_proba, axis=1)
            results["confidence"] = {
                "mean": float(np.mean(max_conf)),
                "median": float(np.median(max_conf)),
                "std": float(np.std(max_conf)),
                "below_0.7": int(np.sum(max_conf < 0.7)),
            }
        
        return results

    def _save_json_report(self, results: Dict[str, Any]) -> None:
        """Save results as structured JSON."""
        path = self.tables_dir / "evaluation_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("JSON report saved: %s", path)
