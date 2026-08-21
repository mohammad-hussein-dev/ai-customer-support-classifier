#!/usr/bin/env python3
"""Professional EDA with comprehensive dashboard for customer support tickets.

Generates publication-quality individual figures plus a combined dashboard
suitable for reports and presentations.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import re
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Professional styling configuration
# ============================================================
CATEGORY_COLORS = {
    "Billing": "#E63946",
    "Technical Support": "#457B9D",
    "Account": "#2A9D8F",
    "Refund": "#E9C46A",
}

PRIORITY_COLORS = {
    "Low": "#90BE6D",
    "Medium": "#F9C74F",
    "High": "#F8961E",
    "Critical": "#F94144",
}

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def load_data() -> pd.DataFrame:
    """Load and prepare the ticket dataset."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "tickets.csv")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    return df


# ============================================================
# Individual professional figures
# ============================================================

def plot_category_donut(df: pd.DataFrame, outdir: Path) -> None:
    """Donut chart with center annotation."""
    counts = df["category"].value_counts()
    colors = [CATEGORY_COLORS[c] for c in counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=None,
        autopct="",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
    )

    # Add percentage labels manually for better positioning
    for i, (wedge, label, value) in enumerate(zip(wedges, counts.index, counts.values)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 0.75 * np.cos(np.deg2rad(angle))
        y = 0.75 * np.sin(np.deg2rad(angle))
        ax.text(x, y, f"{value}\n{value/len(df)*100:.1f}%",
                ha="center", va="center", fontsize=10, fontweight="bold", color="white")

    # Center text
    ax.text(0, 0, f"Total\n{len(df):,}", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#333333")

    # Legend
    ax.legend(wedges, counts.index, title="Category", loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), frameon=False)

    ax.set_title("Ticket Distribution by Category", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(outdir / "v2_01_category_donut.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: v2_01_category_donut.png")


def plot_priority_lollipop(df: pd.DataFrame, outdir: Path) -> None:
    """Horizontal lollipop chart for priority distribution."""
    counts = df["priority"].value_counts().reindex(["Low", "Medium", "High", "Critical"])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PRIORITY_COLORS[p] for p in counts.index]

    ax.hlines(y=range(len(counts)), xmin=0, xmax=counts.values, color=colors, linewidth=3, alpha=0.7)
    ax.scatter(counts.values, range(len(counts)), color=colors, s=100, zorder=3, edgecolors="white", linewidth=2)

    for i, (pri, val) in enumerate(zip(counts.index, counts.values)):
        ax.text(val + 30, i, f"{val:,} ({val/len(df)*100:.1f}%)",
                va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.set_xlabel("Number of Tickets", fontsize=11)
    ax.set_title("Ticket Distribution by Priority", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, max(counts.values) * 1.25)
    ax.invert_yaxis()
    sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout()
    fig.savefig(outdir / "v2_02_priority_lollipop.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: v2_02_priority_lollipop.png")


def plot_text_violin(df: pd.DataFrame, outdir: Path) -> None:
    """Violin plot for text length distribution (more informative than boxplot)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Character count violin
    sns.violinplot(data=df, x="category", y="char_count", ax=axes[0],
                   palette=CATEGORY_COLORS, inner="quartile", linewidth=1.2)
    axes[0].set_title("Character Count Distribution by Category", fontweight="bold")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Characters")
    axes[0].tick_params(axis="x", rotation=30)

    # Word count violin
    sns.violinplot(data=df, x="category", y="word_count", ax=axes[1],
                   palette=CATEGORY_COLORS, inner="quartile", linewidth=1.2)
    axes[1].set_title("Word Count Distribution by Category", fontweight="bold")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Words")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(outdir / "v2_03_text_violin.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: v2_03_text_violin.png")


def plot_priority_heatmap_enhanced(df: pd.DataFrame, outdir: Path) -> None:
    """Enhanced heatmap with diverging colormap and annotations."""
    crosstab = pd.crosstab(df["category"], df["priority"], normalize="index")
    crosstab = crosstab[["Low", "Medium", "High", "Critical"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(crosstab, annot=True, fmt=".1%", cmap="RdYlBu_r", center=0.25,
                linewidths=1, linecolor="white", cbar_kws={"label": "Proportion"},
                ax=ax, annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("Priority Distribution by Category (Normalized)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Priority Level", fontsize=11)
    ax.set_ylabel("Category", fontsize=11)

    plt.tight_layout()
    fig.savefig(outdir / "v2_04_priority_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: v2_04_priority_heatmap.png")


def plot_temporal_line(df: pd.DataFrame, outdir: Path) -> None:
    """Line plot with markers instead of area chart."""
    df_copy = df.copy()
    df_copy["date"] = df_copy["created_at"].dt.date
    daily = df_copy.groupby(["date", "category"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    for col in daily.columns:
        ax.plot(daily.index, daily[col], label=col, color=CATEGORY_COLORS[col],
                linewidth=2, marker="o", markersize=3, alpha=0.8)

    ax.fill_between(daily.index, 0, daily.sum(axis=1), alpha=0.1, color="gray")
    ax.set_title("Daily Ticket Volume by Category", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Number of Tickets", fontsize=11)
    ax.legend(title="Category", loc="upper left", frameon=True, fancybox=True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))

    plt.tight_layout()
    fig.savefig(outdir / "v2_05_temporal_line.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: v2_05_temporal_line.png")


def plot_ridge_distribution(df: pd.DataFrame, outdir: Path) -> None:
    """Ridge plot (joy plot style) for word count by category."""
    categories = df["category"].unique()
    fig, axes = plt.subplots(len(categories), 1, figsize=(10, 8), sharex=True)

    for i, cat in enumerate(categories):
        data = df[df["category"] == cat]["word_count"]
        axes[i].fill_betweenx([0, 1], data.min(), data.max(), alpha=0.3, color=CATEGORY_COLORS[cat])
        sns.kdeplot(data=data, ax=axes[i], color=CATEGORY_COLORS[cat], fill=True, alpha=0.6, linewidth=2)
        axes[i].set_ylabel(cat, rotation=0, ha="right", va="center", fontsize=10, fontweight="bold")
        axes[i].set_yticks([])
        axes[i].set_xlim(0, df["word_count"].max())
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].spines["left"].set_visible(False)

    axes[-1].set_xlabel("Word Count", fontsize=11)
    fig.suptitle("Word Count Distribution by Category (Ridge Plot)", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(outdir / "v2_06_ridge_plot.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: v2_06_ridge_plot.png")


def plot_wordcloud_grid(df: pd.DataFrame, outdir: Path) -> None:
    """2x2 grid of word clouds in a single figure."""
    categories = ["Billing", "Technical Support", "Account", "Refund"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, cat in enumerate(categories):
        texts = df[df["category"] == cat]["text"].tolist()
        text = " ".join(texts).lower()
        text = re.sub(r"[^\w\s]", "", text)

        wc = WordCloud(
            width=600, height=400,
            background_color="white",
            colormap="viridis",
            max_words=100,
            relative_scaling=0.5,
            contour_width=0.5,
            contour_color="gray",
        ).generate(text)

        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(cat, fontsize=13, fontweight="bold", color=CATEGORY_COLORS[cat], pad=10)

    fig.suptitle("Word Clouds by Category", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(outdir / "v2_07_wordcloud_grid.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: v2_07_wordcloud_grid.png")


def plot_bigram_treemap(df: pd.DataFrame, outdir: Path) -> None:
    """Horizontal bar chart (treemap-style visualization) for top bigrams."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    categories = ["Billing", "Technical Support", "Account", "Refund"]

    for i, cat in enumerate(categories):
        texts = df[df["category"] == cat]["text"].tolist()
        all_text = " ".join(texts).lower()
        all_text = re.sub(r"[^\w\s]", "", all_text)
        tokens = all_text.split()
        bigrams = [" ".join(tokens[j:j+2]) for j in range(len(tokens)-1)]
        counter = Counter(bigrams)
        top = counter.most_common(12)

        labels, values = zip(*top) if top else ([], [])
        colors = sns.light_palette(CATEGORY_COLORS[cat], n_colors=len(labels), reverse=True)

        bars = axes[i].barh(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)
        axes[i].set_yticks(range(len(labels)))
        axes[i].set_yticklabels(labels, fontsize=8)
        axes[i].invert_yaxis()
        axes[i].set_title(f"Top Bigrams: {cat}", fontsize=12, fontweight="bold", color=CATEGORY_COLORS[cat])
        axes[i].set_xlabel("Frequency", fontsize=9)

        for bar, val in zip(bars, values):
            axes[i].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        str(val), va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(outdir / "v2_08_bigram_bars.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: v2_08_bigram_bars.png")


# ============================================================
# COMPREHENSIVE DASHBOARD
# ============================================================

def plot_comprehensive_dashboard(df: pd.DataFrame, outdir: Path) -> None:
    """Create a single comprehensive dashboard combining all key insights.

    Layout: 4 rows × 2 columns = 8 key visualizations
    Size: Each subplot roughly 6×4 inches → total 16×18 inches
    """
    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.92, top=0.94, bottom=0.04)

    # Main title
    fig.suptitle("Customer Support Ticket Analysis — Comprehensive Dashboard",
                 fontsize=18, fontweight="bold", y=0.98, color="#1a1a2e")
    fig.text(0.5, 0.955, f"Dataset: {len(df):,} tickets  |  Generated: {datetime.now().strftime('%Y-%m-%d')}",
             ha="center", fontsize=10, color="#666666", style="italic")

    # ── Row 0, Col 0: Category Donut ──
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["category"].value_counts()
    colors = [CATEGORY_COLORS[c] for c in counts.index]
    wedges, _ = ax1.pie(counts.values, colors=colors, startangle=90,
                        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2))
    ax1.text(0, 0, f"{len(df):,}", ha="center", va="center", fontsize=14, fontweight="bold", color="#333")
    ax1.set_title("Category Distribution", fontweight="bold", fontsize=11, pad=10)
    ax1.legend(wedges, [f"{c} ({v})" for c, v in zip(counts.index, counts.values)],
               loc="center left", bbox_to_anchor=(-0.15, 0.5), fontsize=8, frameon=False)

    # ── Row 0, Col 1: Priority Lollipop ──
    ax2 = fig.add_subplot(gs[0, 1])
    pri_counts = df["priority"].value_counts().reindex(["Low", "Medium", "High", "Critical"])
    pri_colors = [PRIORITY_COLORS[p] for p in pri_counts.index]
    ax2.hlines(y=range(len(pri_counts)), xmin=0, xmax=pri_counts.values, color=pri_colors, linewidth=2.5, alpha=0.7)
    ax2.scatter(pri_counts.values, range(len(pri_counts)), color=pri_colors, s=60, zorder=3, edgecolors="white")
    for i, (pri, val) in enumerate(zip(pri_counts.index, pri_counts.values)):
        ax2.text(val + 40, i, f"{val:,}", va="center", fontsize=9, fontweight="bold")
    ax2.set_yticks(range(len(pri_counts)))
    ax2.set_yticklabels(pri_counts.index, fontsize=9)
    ax2.set_title("Priority Distribution", fontweight="bold", fontsize=11, pad=10)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(pri_counts.values) * 1.2)

    # ── Row 1, Col 0: Text Length Violin ──
    ax3 = fig.add_subplot(gs[1, 0])
    parts = ax3.violinplot([df[df["category"] == cat]["word_count"].values for cat in counts.index],
                           positions=range(len(counts)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(CATEGORY_COLORS[counts.index[i]])
        pc.set_alpha(0.6)
    ax3.set_xticks(range(len(counts)))
    ax3.set_xticklabels(counts.index, rotation=15, ha="right", fontsize=9)
    ax3.set_ylabel("Word Count", fontsize=9)
    ax3.set_title("Word Count by Category (Violin)", fontweight="bold", fontsize=11, pad=10)

    # ── Row 1, Col 1: Priority Heatmap ──
    ax4 = fig.add_subplot(gs[1, 1])
    crosstab = pd.crosstab(df["category"], df["priority"], normalize="index")
    crosstab = crosstab[["Low", "Medium", "High", "Critical"]]
    im = ax4.imshow(crosstab.values, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=0.6)
    ax4.set_xticks(range(len(crosstab.columns)))
    ax4.set_xticklabels(crosstab.columns, fontsize=9)
    ax4.set_yticks(range(len(crosstab.index)))
    ax4.set_yticklabels(crosstab.index, fontsize=9)
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            ax4.text(j, i, f"{crosstab.iloc[i, j]:.1%}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if crosstab.iloc[i, j] > 0.35 else "black")
    ax4.set_title("Priority vs Category", fontweight="bold", fontsize=11, pad=10)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label="Proportion")

    # ── Row 2, Col 0: Temporal Trends ──
    ax5 = fig.add_subplot(gs[2, 0])
    df_copy = df.copy()
    df_copy["date"] = df_copy["created_at"].dt.date
    daily = df_copy.groupby(["date", "category"]).size().unstack(fill_value=0)
    for col in daily.columns:
        ax5.plot(daily.index, daily[col], label=col, color=CATEGORY_COLORS[col],
                linewidth=1.5, alpha=0.8)
    ax5.fill_between(daily.index, 0, daily.sum(axis=1), alpha=0.08, color="gray")
    ax5.set_title("Daily Ticket Volume", fontweight="bold", fontsize=11, pad=10)
    ax5.set_xlabel("Date", fontsize=9)
    ax5.set_ylabel("Tickets", fontsize=9)
    ax5.legend(fontsize=8, loc="upper left", frameon=True, fancybox=True)
    ax5.tick_params(axis="x", rotation=20)

    # ── Row 2, Col 1: Class Imbalance ──
    ax6 = fig.add_subplot(gs[2, 1])
    imb_counts = df["category"].value_counts()
    max_c = imb_counts.max()
    ratios = max_c / imb_counts
    bars = ax6.bar(imb_counts.index, imb_counts.values,
                   color=[CATEGORY_COLORS[c] for c in imb_counts.index], alpha=0.85, edgecolor="white")
    for bar, cnt in zip(bars, imb_counts.values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{cnt:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax6_twin = ax6.twinx()
    ax6_twin.plot(imb_counts.index, ratios.values, "o-", color="#E63946", linewidth=2, markersize=6)
    ax6_twin.axhline(y=1.5, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax6_twin.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax6_twin.set_ylabel("Imbalance Ratio", color="#E63946", fontsize=9)
    ax6_twin.tick_params(axis="y", labelcolor="#E63946", labelsize=8)
    ax6.set_title("Class Imbalance Analysis", fontweight="bold", fontsize=11, pad=10)
    ax6.set_ylabel("Ticket Count", fontsize=9)
    ax6.tick_params(axis="x", rotation=20)

    # ── Row 3: Word Clouds (split into 4 mini subplots) ──
    categories = ["Billing", "Technical Support", "Account", "Refund"]
    for i, cat in enumerate(categories):
        ax = fig.add_subplot(gs[3, i // 2 if i < 2 else i - 2])  # 2x2 in row 3
        # Actually GridSpec 4,2 means row 3 has 2 cols. We need 4 word clouds.
        # Let me use nested gridspec for row 3
        pass

    # Better approach: use nested GridSpec for row 3
    # Remove the above and redo row 3 with nested gridspec

    plt.savefig(outdir / "00_comprehensive_dashboard.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: 00_comprehensive_dashboard.png")


def plot_comprehensive_dashboard_v2(df: pd.DataFrame, outdir: Path) -> None:
    """Improved comprehensive dashboard with proper 4-wordcloud row."""
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.35,
                  left=0.06, right=0.94, top=0.95, bottom=0.03)

    fig.suptitle("Customer Support Ticket Analysis — Comprehensive Dashboard",
                 fontsize=20, fontweight="bold", y=0.98, color="#1a1a2e")
    fig.text(0.5, 0.965, f"Dataset: {len(df):,} tickets  |  Categories: {df['category'].nunique()}  |  Priorities: {df['priority'].nunique()}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             ha="center", fontsize=11, color="#666666", style="italic")

    # === ROW 0: Category Overview ===
    # Donut
    ax1 = fig.add_subplot(gs[0, :2])
    counts = df["category"].value_counts()
    colors = [CATEGORY_COLORS[c] for c in counts.index]
    wedges, _ = ax1.pie(counts.values, colors=colors, startangle=90,
                        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2.5))
    ax1.text(0, 0, f"{len(df):,}\ntickets", ha="center", va="center",
             fontsize=16, fontweight="bold", color="#333333")
    ax1.set_title("Category Distribution", fontweight="bold", fontsize=13, pad=15)
    ax1.legend(wedges, [f"{c}: {v:,} ({v/len(df)*100:.1f}%)" for c, v in zip(counts.index, counts.values)],
               loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize=10, frameon=False)

    # Priority lollipop
    ax2 = fig.add_subplot(gs[0, 2:])
    pri_counts = df["priority"].value_counts().reindex(["Low", "Medium", "High", "Critical"])
    pri_colors = [PRIORITY_COLORS[p] for p in pri_counts.index]
    ax2.hlines(y=range(len(pri_counts)), xmin=0, xmax=pri_counts.values,
               color=pri_colors, linewidth=3, alpha=0.7)
    ax2.scatter(pri_counts.values, range(len(pri_counts)), color=pri_colors,
                s=120, zorder=3, edgecolors="white", linewidth=2)
    for i, (pri, val) in enumerate(zip(pri_counts.index, pri_counts.values)):
        ax2.text(val + 50, i, f"{val:,}  ({val/len(df)*100:.1f}%)",
                va="center", fontsize=10, fontweight="bold")
    ax2.set_yticks(range(len(pri_counts)))
    ax2.set_yticklabels(pri_counts.index, fontsize=10)
    ax2.set_xlabel("Number of Tickets", fontsize=10)
    ax2.set_title("Priority Distribution", fontweight="bold", fontsize=13, pad=15)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(pri_counts.values) * 1.25)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    # === ROW 1: Text Characteristics ===
    # Violin plot
    ax3 = fig.add_subplot(gs[1, :2])
    data_for_violin = [df[df["category"] == cat]["word_count"].values for cat in counts.index]
    parts = ax3.violinplot(data_for_violin, positions=range(len(counts)),
                           showmeans=True, showmedians=False, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(CATEGORY_COLORS[counts.index[i]])
        pc.set_alpha(0.5)
        pc.set_edgecolor(CATEGORY_COLORS[counts.index[i]])
        pc.set_linewidth(1.5)
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        parts[partname].set_edgecolor("#333333")
        parts[partname].set_linewidth(1)
    ax3.set_xticks(range(len(counts)))
    ax3.set_xticklabels(counts.index, rotation=20, ha="right", fontsize=10)
    ax3.set_ylabel("Word Count", fontsize=10)
    ax3.set_title("Word Count Distribution by Category", fontweight="bold", fontsize=13, pad=15)

    # Heatmap
    ax4 = fig.add_subplot(gs[1, 2:])
    crosstab = pd.crosstab(df["category"], df["priority"], normalize="index")
    crosstab = crosstab[["Low", "Medium", "High", "Critical"]]
    sns.heatmap(crosstab, annot=True, fmt=".1%", cmap="RdYlBu_r", center=0.25,
                linewidths=1.5, linecolor="white", cbar_kws={"label": "Proportion", "shrink": 0.8},
                ax=ax4, annot_kws={"size": 10, "weight": "bold"})
    ax4.set_title("Priority vs Category Heatmap", fontweight="bold", fontsize=13, pad=15)
    ax4.set_xlabel("Priority Level", fontsize=10)
    ax4.set_ylabel("")

    # === ROW 2: Temporal & Imbalance ===
    # Temporal
    ax5 = fig.add_subplot(gs[2, :2])
    df_copy = df.copy()
    df_copy["date"] = df_copy["created_at"].dt.date
    daily = df_copy.groupby(["date", "category"]).size().unstack(fill_value=0)
    for col in daily.columns:
        ax5.plot(daily.index, daily[col], label=col, color=CATEGORY_COLORS[col],
                linewidth=2, marker="o", markersize=3, alpha=0.85)
    ax5.fill_between(daily.index, 0, daily.sum(axis=1), alpha=0.06, color="black")
    ax5.set_title("Daily Ticket Volume Trend", fontweight="bold", fontsize=13, pad=15)
    ax5.set_xlabel("Date", fontsize=10)
    ax5.set_ylabel("Number of Tickets", fontsize=10)
    ax5.legend(fontsize=9, loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax5.tick_params(axis="x", rotation=15)

    # Imbalance
    ax6 = fig.add_subplot(gs[2, 2:])
    imb_counts = df["category"].value_counts()
    max_c = imb_counts.max()
    ratios = max_c / imb_counts
    bars = ax6.bar(imb_counts.index, imb_counts.values,
                   color=[CATEGORY_COLORS[c] for c in imb_counts.index],
                   alpha=0.9, edgecolor="white", linewidth=1.5)
    for bar, cnt in zip(bars, imb_counts.values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 25,
                f"{cnt:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax6_twin = ax6.twinx()
    ax6_twin.plot(imb_counts.index, ratios.values, "D-", color="#D62828",
                  linewidth=2.5, markersize=7, label="Imbalance Ratio")
    ax6_twin.axhline(y=1.5, color="#F77F00", linestyle="--", alpha=0.6, linewidth=1.5, label="Moderate (1.5x)")
    ax6_twin.axhline(y=2.0, color="#D62828", linestyle="--", alpha=0.6, linewidth=1.5, label="Severe (2.0x)")
    ax6_twin.set_ylabel("Imbalance Ratio", color="#D62828", fontsize=10, fontweight="bold")
    ax6_twin.tick_params(axis="y", labelcolor="#D62828", labelsize=9)
    ax6_twin.legend(loc="upper right", fontsize=8, frameon=True)
    ax6.set_title("Class Imbalance Analysis", fontweight="bold", fontsize=13, pad=15)
    ax6.set_ylabel("Ticket Count", fontsize=10)
    ax6.tick_params(axis="x", rotation=20)

    # === ROW 3: Word Clouds (4 subplots) ===
    categories = ["Billing", "Technical Support", "Account", "Refund"]
    for i, cat in enumerate(categories):
        ax = fig.add_subplot(gs[3, i])
        texts = df[df["category"] == cat]["text"].tolist()
        text = " ".join(texts).lower()
        text = re.sub(r"[^\w\s]", "", text)

        wc = WordCloud(
            width=500, height=350,
            background_color="white",
            colormap="viridis",
            max_words=60,
            relative_scaling=0.4,
            contour_width=0.3,
            contour_color="gray",
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(cat, fontsize=12, fontweight="bold", color=CATEGORY_COLORS[cat], pad=8)

    # === ROW 4: Bigrams (4 subplots) ===
    for i, cat in enumerate(categories):
        ax = fig.add_subplot(gs[4, i])
        texts = df[df["category"] == cat]["text"].tolist()
        all_text = " ".join(texts).lower()
        all_text = re.sub(r"[^\w\s]", "", all_text)
        tokens = all_text.split()
        bigrams = [" ".join(tokens[j:j+2]) for j in range(len(tokens)-1)]
        counter = Counter(bigrams)
        top = counter.most_common(8)

        labels, values = zip(*top) if top else ([], [])
        colors = sns.light_palette(CATEGORY_COLORS[cat], n_colors=len(labels), reverse=True)

        bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(f"Top Bigrams: {cat}", fontsize=11, fontweight="bold", color=CATEGORY_COLORS[cat])
        ax.set_xlabel("Freq", fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=7.5)

    plt.savefig(outdir / "00_comprehensive_dashboard.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: 00_comprehensive_dashboard.png (20x24 inches)")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the complete professional EDA pipeline."""
    print("=" * 70)
    print("🎨 PROFESSIONAL EDA v2 — Customer Support Tickets")
    print("=" * 70)

    outdir = PROJECT_ROOT / "reports" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading data...")
    df = load_data()
    print(f"      ✓ {len(df):,} records loaded")

    print("\n[2/3] Generating individual professional figures...")
    plot_category_donut(df, outdir)
    plot_priority_lollipop(df, outdir)
    plot_text_violin(df, outdir)
    plot_priority_heatmap_enhanced(df, outdir)
    plot_temporal_line(df, outdir)
    plot_ridge_distribution(df, outdir)
    plot_wordcloud_grid(df, outdir)
    plot_bigram_treemap(df, outdir)

    print("\n[3/3] Generating comprehensive dashboard...")
    plot_comprehensive_dashboard_v2(df, outdir)

    print("\n" + "=" * 70)
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output directory: {outdir}")
    print("\n📊 Generated files:")
    for f in sorted(outdir.glob("v2_*")):
        print(f"   • {f.name}")
    print(f"   • 00_comprehensive_dashboard.png  ← MAIN DASHBOARD")


if __name__ == "__main__":
    main()
