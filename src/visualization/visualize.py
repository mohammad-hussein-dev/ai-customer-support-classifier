"""
Professional EDA & Evaluation Visualization Engine
====================================================
Dark-theme, publication-quality figures for banking intent classification.
Supports combined dashboards and individual plots.

Author: Mohammad Hussein
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DARK THEME CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DARK_BG = "#0d1117"
DARK_CARD = "#161b22"
DARK_BORDER = "#30363d"
TEXT_PRIMARY = "#c9d1d9"
TEXT_SECONDARY = "#8b949e"
ACCENT_CYAN = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_YELLOW = "#d29922"
ACCENT_RED = "#f85149"
ACCENT_PURPLE = "#a371f7"
ACCENT_PINK = "#f778ba"

GRADIENT_PALETTE = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#a371f7", "#f778ba", "#56d364", "#79c0ff"]


def _setup_dark_theme():
    """Configure matplotlib for dark theme."""
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_CARD,
        "axes.edgecolor": DARK_BORDER,
        "axes.labelcolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "grid.color": DARK_BORDER,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.facecolor": DARK_BG,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Noto Sans", "Arial"],
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


_setup_dark_theme()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VISUALIZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class BankingVisualizer:
    """Professional visualization engine for banking intent classification.
    
    Produces dark-theme, publication-quality figures suitable for:
    - EDA reports
    - Model evaluation
    - Executive dashboards
    - Academic papers
    
    Attributes:
        output_dir: Directory to save figures.
        palette: Color palette for plots.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "reports/figures",
        palette: List[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.palette = palette or GRADIENT_PALETTE
        _setup_dark_theme()
        logger.info("BankingVisualizer initialized | output=%s", self.output_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 1: COMBINED EDA DASHBOARD (The "One Figure")
    # ─────────────────────────────────────────────────────────────────────────

    def plot_eda_dashboard(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "intent",
        save_name: str = "00_eda_master_dashboard.png",
    ) -> Path:
        """Generate a single comprehensive EDA dashboard (2x3 grid).
        
        Combines: intent distribution, char length violin, word count box,
        class imbalance, avg length by intent, sample count by intent.
        """
        df = df.copy()
        df["char_count"] = df[text_col].str.len()
        df["word_count"] = df[text_col].str.split().str.len()
        
        intents = df[label_col].value_counts().index.tolist()
        n_intents = len(intents)
        colors = self.palette[:n_intents]

        fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # ── Title ──
        fig.suptitle(
            "Banking Intent Classification — Exploratory Data Analysis",
            fontsize=18, fontweight="bold", color=TEXT_PRIMARY, y=0.98,
        )
        fig.text(0.5, 0.955, f"Dataset: {len(df):,} tickets  |  Intents: {n_intents}  |  Avg length: {df['char_count'].mean():.0f} chars",
                 ha="center", fontsize=10, color=TEXT_SECONDARY)

        # ── 1. Intent Distribution (Horizontal Bar) ──
        ax1 = fig.add_subplot(gs[0, :2])
        counts = df[label_col].value_counts()
        bars = ax1.barh(range(len(counts)), counts.values, color=colors, height=0.7, edgecolor=DARK_BORDER, linewidth=0.5)
        ax1.set_yticks(range(len(counts)))
        ax1.set_yticklabels([i.replace("_", " ").title() for i in counts.index])
        ax1.invert_yaxis()
        ax1.set_xlabel("Number of Tickets", color=TEXT_SECONDARY)
        ax1.set_title("📊 Intent Distribution", fontweight="bold", pad=10, color=ACCENT_CYAN)
        
        for i, (bar, val) in enumerate(zip(bars, counts.values)):
            pct = val / len(df) * 100
            ax1.text(val + max(counts.values)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:,}  ({pct:.1f}%)", va="center", ha="left", fontsize=9,
                    color=TEXT_PRIMARY, fontweight="bold")
        ax1.set_xlim(0, max(counts.values) * 1.25)

        # ── 2. Class Imbalance Ratio ──
        ax2 = fig.add_subplot(gs[0, 2])
        max_c = counts.max()
        ratios = max_c / counts
        bars2 = ax2.bar(range(len(ratios)), ratios.values, color=[ACCENT_RED if r > 2 else ACCENT_YELLOW if r > 1.5 else ACCENT_GREEN for r in ratios.values],
                       edgecolor=DARK_BORDER, linewidth=0.5)
        ax2.set_xticks(range(len(ratios)))
        ax2.set_xticklabels([i.replace("_", " ").title()[:12] for i in ratios.index], rotation=45, ha="right", fontsize=8)
        ax2.axhline(y=1.5, color=ACCENT_YELLOW, linestyle="--", alpha=0.7, linewidth=1)
        ax2.axhline(y=2.0, color=ACCENT_RED, linestyle="--", alpha=0.7, linewidth=1)
        ax2.set_title("⚖️ Imbalance Ratio", fontweight="bold", pad=10, color=ACCENT_YELLOW)
        ax2.set_ylabel("Majority / Class", color=TEXT_SECONDARY)
        for bar, val in zip(bars2, ratios.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{val:.1f}x", ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY, fontweight="bold")

        # ── 3. Character Length Violin ──
        ax3 = fig.add_subplot(gs[1, 0])
        parts = ax3.violinplot([df[df[label_col] == intent]["char_count"].values for intent in intents],
                               positions=range(n_intents), showmeans=True, showmedians=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor(colors[i])
        parts["cmeans"].set_color(TEXT_PRIMARY)
        parts["cmeans"].set_linewidth(1.5)
        ax3.set_xticks(range(n_intents))
        ax3.set_xticklabels([i.replace("_", " ").title()[:10] for i in intents], rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Characters", color=TEXT_SECONDARY)
        ax3.set_title("📝 Character Length", fontweight="bold", pad=10, color=ACCENT_GREEN)

        # ── 4. Word Count Box Plot ──
        ax4 = fig.add_subplot(gs[1, 1])
        bp = ax4.boxplot([df[df[label_col] == intent]["word_count"].values for intent in intents],
                         patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor(color)
        for whisker in bp["whiskers"]:
            whisker.set_color(TEXT_SECONDARY)
            whisker.set_linewidth(1)
        for cap in bp["caps"]:
            cap.set_color(TEXT_SECONDARY)
        for median in bp["medians"]:
            median.set_color(TEXT_PRIMARY)
            median.set_linewidth(2)
        ax4.set_xticklabels([i.replace("_", " ").title()[:10] for i in intents], rotation=45, ha="right", fontsize=8)
        ax4.set_ylabel("Words", color=TEXT_SECONDARY)
        ax4.set_title("🔤 Word Count Distribution", fontweight="bold", pad=10, color=ACCENT_PURPLE)

        # ── 5. Average Length by Intent ──
        ax5 = fig.add_subplot(gs[1, 2])
        avg_chars = df.groupby(label_col)["char_count"].mean().reindex(intents)
        avg_words = df.groupby(label_col)["word_count"].mean().reindex(intents)
        x = np.arange(n_intents)
        w = 0.35
        bars_c = ax5.bar(x - w/2, avg_chars.values, w, label="Chars", color=ACCENT_CYAN, alpha=0.8, edgecolor=DARK_BORDER)
        bars_w = ax5.bar(x + w/2, avg_words.values, w, label="Words", color=ACCENT_PINK, alpha=0.8, edgecolor=DARK_BORDER)
        ax5.set_xticks(x)
        ax5.set_xticklabels([i.replace("_", " ").title()[:10] for i in intents], rotation=45, ha="right", fontsize=8)
        ax5.set_title("📏 Avg Length by Intent", fontweight="bold", pad=10, color=ACCENT_CYAN)
        ax5.legend(loc="upper right", framealpha=0.3)
        for bar in bars_c:
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7, color=TEXT_PRIMARY)

        # ── 6. Length Difference Significance (KDE overlay) ──
        ax6 = fig.add_subplot(gs[2, :])
        # Show KDE for top 3 most common intents
        top3 = counts.head(3).index.tolist()
        for intent, color in zip(top3, [ACCENT_CYAN, ACCENT_GREEN, ACCENT_YELLOW]):
            data = df[df[label_col] == intent]["word_count"].values
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax6.fill_between(x_range, kde(x_range), alpha=0.3, color=color, label=intent.replace("_", " ").title())
            ax6.plot(x_range, kde(x_range), color=color, linewidth=2)
        ax6.set_xlabel("Word Count", color=TEXT_SECONDARY)
        ax6.set_ylabel("Density", color=TEXT_SECONDARY)
        ax6.set_title("📈 Message Length Density (Top 3 Intents)", fontweight="bold", pad=10, color=ACCENT_GREEN)
        ax6.legend(loc="upper right", framealpha=0.3)

        # ── Footer ──
        fig.text(0.99, 0.01, "Generated by BankingVisualizer | Dark Theme v2.0", ha="right", fontsize=7,
                color=TEXT_SECONDARY, alpha=0.5)

        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches="tight", facecolor=DARK_BG, edgecolor="none")
        plt.close(fig)
        logger.info("✅ EDA Dashboard saved: %s", save_path)
        return save_path

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 2: CONFUSION MATRIX + PER-CLASS METRICS (Combined)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_evaluation_dashboard(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        y_proba: Optional[np.ndarray] = None,
        save_name: str = "01_evaluation_dashboard.png",
    ) -> Path:
        """Combined evaluation figure: confusion matrix + per-class metrics + confidence."""
        
        # Compute metrics
        f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
        precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        fig.suptitle(
            f"Model Evaluation Dashboard  |  Macro-F1: {macro_f1:.4f}",
            fontsize=18, fontweight="bold", color=TEXT_PRIMARY, y=0.98,
        )

        # ── 1. Confusion Matrix (Normalized) ──
        ax1 = fig.add_subplot(gs[:, :2])
        im = ax1.imshow(cm_norm, cmap="magma", aspect="auto", vmin=0, vmax=1)
        
        # Annotate
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_norm[i, j]
                color = "white" if val > 0.5 else TEXT_PRIMARY
                if val > 0.01:
                    ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")
                # Add raw count in smaller text
                ax1.text(j, i + 0.25, f"({cm[i,j]})", ha="center", va="center", color=color, fontsize=7, alpha=0.7)

        ax1.set_xticks(range(len(class_names)))
        ax1.set_yticks(range(len(class_names)))
        ax1.set_xticklabels([c.replace("_", "\\n").title() for c in class_names], fontsize=8, rotation=0)
        ax1.set_yticklabels([c.replace("_", " ").title() for c in class_names], fontsize=8)
        ax1.set_xlabel("Predicted Intent", color=TEXT_SECONDARY, fontsize=11)
        ax1.set_ylabel("True Intent", color=TEXT_SECONDARY, fontsize=11)
        ax1.set_title("🔥 Normalized Confusion Matrix", fontweight="bold", pad=10, color=ACCENT_CYAN)
        
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("Proportion", color=TEXT_SECONDARY)
        cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY)

        # ── 2. Per-Class F1 Scores ──
        ax2 = fig.add_subplot(gs[0, 2])
        sorted_idx = np.argsort(f1s)[::-1]
        sorted_names = [class_names[i].replace("_", " ").title() for i in sorted_idx]
        sorted_f1s = f1s[sorted_idx]
        colors_bar = [ACCENT_GREEN if f >= 0.9 else ACCENT_YELLOW if f >= 0.75 else ACCENT_RED for f in sorted_f1s]
        
        bars = ax2.barh(range(len(sorted_f1s)), sorted_f1s, color=colors_bar, height=0.6, edgecolor=DARK_BORDER)
        ax2.set_yticks(range(len(sorted_f1s)))
        ax2.set_yticklabels(sorted_names, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 1.05)
        ax2.axvline(x=macro_f1, color=ACCENT_CYAN, linestyle="--", linewidth=2, label=f"Macro-F1: {macro_f1:.3f}")
        ax2.set_xlabel("F1-Score", color=TEXT_SECONDARY)
        ax2.set_title("📊 Per-Class F1 (Sorted)", fontweight="bold", pad=10, color=ACCENT_GREEN)
        ax2.legend(loc="lower right", framealpha=0.3)
        
        for bar, val in zip(bars, sorted_f1s):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                    va="center", ha="left", fontsize=9, color=TEXT_PRIMARY, fontweight="bold")

        # ── 3. Precision vs Recall Scatter ──
        ax3 = fig.add_subplot(gs[1, 2])
        scatter_colors = [ACCENT_GREEN if f >= 0.9 else ACCENT_YELLOW if f >= 0.75 else ACCENT_RED for f in f1s]
        ax3.scatter(recalls, precisions, c=scatter_colors, s=120, edgecolors=TEXT_PRIMARY, linewidth=1.5, alpha=0.9, zorder=3)
        
        for i, name in enumerate(class_names):
            ax3.annotate(name.replace("_", " ").title()[:12], (recalls[i], precisions[i]),
                        textcoords="offset points", xytext=(8, 5), fontsize=7, color=TEXT_SECONDARY)
        
        ax3.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=1)
        ax3.set_xlim(0, 1.05)
        ax3.set_ylim(0, 1.05)
        ax3.set_xlabel("Recall", color=TEXT_SECONDARY)
        ax3.set_ylabel("Precision", color=TEXT_SECONDARY)
        ax3.set_title("🎯 Precision vs Recall", fontweight="bold", pad=10, color=ACCENT_PURPLE)
        ax3.grid(True, alpha=0.2)

        fig.text(0.99, 0.01, "Generated by BankingVisualizer | Dark Theme v2.0", ha="right", fontsize=7,
                color=TEXT_SECONDARY, alpha=0.5)

        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches="tight", facecolor=DARK_BG, edgecolor="none")
        plt.close(fig)
        logger.info("✅ Evaluation Dashboard saved: %s", save_path)
        return save_path

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 3: TOP TF-IDF TERMS BY INTENT
    # ─────────────────────────────────────────────────────────────────────────

    def plot_tfidf_terms(
        self,
        vectorizer: TfidfVectorizer,
        model_coefs: np.ndarray,
        class_names: List[str],
        top_n: int = 10,
        save_name: str = "02_tfidf_top_terms.png",
    ) -> Path:
        """Plot top TF-IDF terms per intent (Logistic Regression coefficients)."""
        feature_names = vectorizer.get_feature_names_out()
        n_classes = len(class_names)
        n_cols = min(4, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5), facecolor=DARK_BG)
        if n_classes == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        fig.suptitle("Top TF-IDF Terms by Intent (Logistic Regression Coefficients)",
                    fontsize=16, fontweight="bold", color=TEXT_PRIMARY, y=0.98)

        for idx, (intent, ax) in enumerate(zip(class_names, axes)):
            coef = model_coefs[idx]
            top_idx = np.argsort(coef)[-top_n:][::-1]
            top_terms = [feature_names[i] for i in top_idx]
            top_vals = coef[top_idx]

            colors = [ACCENT_GREEN if v > 0 else ACCENT_RED for v in top_vals]
            bars = ax.barh(range(len(top_terms)), top_vals, color=colors, height=0.6, edgecolor=DARK_BORDER)
            ax.set_yticks(range(len(top_terms)))
            ax.set_yticklabels(top_terms, fontsize=9)
            ax.invert_yaxis()
            ax.set_title(intent.replace("_", " ").title(), fontsize=11, fontweight="bold", color=self.palette[idx % len(self.palette)])
            ax.set_xlabel("Coefficient", fontsize=9, color=TEXT_SECONDARY)
            ax.tick_params(colors=TEXT_SECONDARY)
            ax.set_facecolor(DARK_CARD)

            for bar, val in zip(bars, top_vals):
                ax.text(val + 0.01 if val > 0 else val - 0.01, bar.get_y() + bar.get_height()/2,
                       f"{val:.2f}", ha="left" if val > 0 else "right", va="center", fontsize=8, color=TEXT_PRIMARY)

        # Hide unused subplots
        for idx in range(n_classes, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches="tight", facecolor=DARK_BG, edgecolor="none")
        plt.close(fig)
        logger.info("✅ TF-IDF terms plot saved: %s", save_path)
        return save_path

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 4: CONFIDENCE DISTRIBUTION + CALIBRATION
    # ─────────────────────────────────────────────────────────────────────────

    def plot_confidence_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        review_threshold: float = 0.7,
        save_name: str = "03_confidence_analysis.png",
    ) -> Path:
        """Plot confidence distribution and calibration analysis."""
        max_conf = np.max(y_proba, axis=1)
        correct = (y_true == y_pred)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=DARK_BG)
        fig.suptitle("Prediction Confidence & Calibration Analysis", fontsize=16, fontweight="bold", color=TEXT_PRIMARY)

        # ── Histogram: Correct vs Incorrect ──
        ax1 = axes[0]
        bins = np.linspace(0, 1, 21)
        ax1.hist(max_conf[correct], bins=bins, alpha=0.7, label="Correct", color=ACCENT_GREEN, edgecolor=DARK_BORDER)
        ax1.hist(max_conf[~correct], bins=bins, alpha=0.7, label="Incorrect", color=ACCENT_RED, edgecolor=DARK_BORDER)
        ax1.axvline(x=review_threshold, color=ACCENT_YELLOW, linestyle="--", linewidth=2, label=f"Review Threshold ({review_threshold})")
        ax1.axvline(x=np.mean(max_conf), color=ACCENT_CYAN, linestyle="-", linewidth=2, label=f"Mean ({np.mean(max_conf):.3f})")
        ax1.set_xlabel("Confidence", color=TEXT_SECONDARY)
        ax1.set_ylabel("Count", color=TEXT_SECONDARY)
        ax1.set_title("📊 Confidence Distribution", fontweight="bold", color=ACCENT_CYAN)
        ax1.legend(framealpha=0.3)
        ax1.set_facecolor(DARK_CARD)

        # ── Box plot by class ──
        ax2 = axes[1]
        conf_by_class = [max_conf[y_true == cls] for cls in class_names]
        bp = ax2.boxplot(conf_by_class, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], self.palette[:len(class_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor(color)
        for median in bp["medians"]:
            median.set_color(TEXT_PRIMARY)
            median.set_linewidth(2)
        ax2.set_xticklabels([c.replace("_", "\\n").title() for c in class_names], fontsize=8, rotation=0)
        ax2.set_ylabel("Confidence", color=TEXT_SECONDARY)
        ax2.set_title("📦 Confidence by True Intent", fontweight="bold", color=ACCENT_PURPLE)
        ax2.set_facecolor(DARK_CARD)
        ax2.axhline(y=review_threshold, color=ACCENT_YELLOW, linestyle="--", alpha=0.7)

        # ── Reliability diagram (calibration) ──
        ax3 = axes[2]
        from sklearn.calibration import calibration_curve
        # Binary: correct or not
        prob_true, prob_pred = calibration_curve(correct.astype(int), max_conf, n_bins=10)
        ax3.plot(prob_pred, prob_true, "o-", color=ACCENT_CYAN, linewidth=2, markersize=8, label="Model")
        ax3.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfectly calibrated")
        ax3.fill_between(prob_pred, prob_true, alpha=0.2, color=ACCENT_CYAN)
        ax3.set_xlabel("Mean Predicted Confidence", color=TEXT_SECONDARY)
        ax3.set_ylabel("Fraction of Correct Predictions", color=TEXT_SECONDARY)
        ax3.set_title("🎯 Calibration Curve", fontweight="bold", color=ACCENT_GREEN)
        ax3.legend(framealpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_facecolor(DARK_CARD)
        ax3.grid(True, alpha=0.2)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches="tight", facecolor=DARK_BG, edgecolor="none")
        plt.close(fig)
        logger.info("✅ Confidence analysis saved: %s", save_path)
        return save_path

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 5: ERROR ANALYSIS — MISCLASSIFIED EXAMPLES
    # ─────────────────────────────────────────────────────────────────────────

    def plot_error_patterns(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        texts: List[str],
        class_names: List[str],
        y_proba: np.ndarray,
        top_n_pairs: int = 6,
        save_name: str = "04_error_patterns.png",
    ) -> Path:
        """Visualize top confused intent pairs with example texts."""
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        np.fill_diagonal(cm, 0)  # Remove correct predictions

        # Find top confused pairs
        pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    pairs.append((class_names[i], class_names[j], cm[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:top_n_pairs]

        n_pairs = len(top_pairs)
        fig, axes = plt.subplots((n_pairs + 1) // 2, 2, figsize=(16, n_pairs * 2.5), facecolor=DARK_BG)
        if n_pairs == 1:
            axes = np.array([[axes]])
        axes = axes.flatten()

        fig.suptitle("Top Misclassification Patterns with Example Texts", fontsize=16, fontweight="bold", color=TEXT_PRIMARY)

        for idx, (true_lbl, pred_lbl, count) in enumerate(top_pairs):
            ax = axes[idx]
            ax.set_facecolor(DARK_CARD)

            # Find example texts for this error
            mask = (y_true == true_lbl) & (y_pred == pred_lbl)
            example_texts = [t for t, m in zip(texts, mask) if m]
            confidences = [np.max(y_proba[i]) for i, m in enumerate(mask) if m]

            # Title
            ax.text(0.5, 0.9, f"{true_lbl.replace('_', ' ').title()} → {pred_lbl.replace('_', ' ').title()}",
                   ha="center", va="top", fontsize=12, fontweight="bold", color=ACCENT_RED,
                   transform=ax.transAxes)
            ax.text(0.5, 0.78, f"{count} cases  |  Avg confidence: {np.mean(confidences):.3f}",
                   ha="center", va="top", fontsize=9, color=TEXT_SECONDARY,
                   transform=ax.transAxes)

            # Show up to 3 example texts
            for i, txt in enumerate(example_texts[:3]):
                display_txt = txt[:80] + "..." if len(txt) > 80 else txt
                ax.text(0.05, 0.55 - i*0.18, f"• {display_txt}",
                       ha="left", va="top", fontsize=8, color=TEXT_PRIMARY,
                       transform=ax.transAxes, wrap=True,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, edgecolor=DARK_BORDER, alpha=0.8))

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches="tight", facecolor=DARK_BG, edgecolor="none")
        plt.close(fig)
        logger.info("✅ Error patterns saved: %s", save_path)
        return save_path

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 6: KEYWORD BASELINE vs ML COMPARISON
    # ─────────────────────────────────────────────────────────────────────────

    def plot_baseline_comparison(
        self,
        baseline_f1: float,
        ml_f1: float,
        baseline_name: str = "Keyword Baseline",
        ml_name: str = "TF-IDF + Logistic Regression",
        save_name: str = "05_baseline_comparison.png",
    ) -> Path:
        """Compare keyword baseline against ML model."""
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
        ax.set_facecolor(DARK_CARD)

        models = [baseline_name, ml_name]
        scores = [baseline_f1, ml_f1]
        colors = [ACCENT_YELLOW, ACCENT_GREEN]
        
        bars = ax.bar(models, scores, color=colors, width=0.5, edgecolor=DARK_BORDER, linewidth=1.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro-F1 Score", color=TEXT_SECONDARY, fontsize=12)
        ax.set_title("📈 Baseline vs ML Model Performance", fontsize=14, fontweight="bold", color=ACCENT_CYAN)

        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{score:.4f}", ha="center", va="bottom", fontsize=14, fontweight="bold", color=TEXT_PRIMARY)

        # Improvement arrow
        if ml_f1 > baseline_f1:
            improvement = ((ml_f1 - baseline_f1) / baseline_f1) * 100
            ax.annotate(f"+{improvement:.1f}% improvement",
                       xy=(1, ml_f1), xytext=(0.5, (baseline_f1 + ml_f1)/2),
                       arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=2),
                       fontsize=11, color=ACCENT_GREEN, fontweight="bold", ha="center")

        ax.tick_params(colors=TEXT_SECONDARY)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches="tight", facecolor=DARK_BG, edgecolor="none")
        plt.close(fig)
        logger.info("✅ Baseline comparison saved: %s", save_path)
        return save_path
