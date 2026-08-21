"""Exploratory Data Analysis (EDA) visualizations for support tickets.

Generates publication-quality figures for understanding ticket
distributions, text characteristics, and temporal patterns.

Example:
    >>> from src.visualization.visualize import TicketVisualizer
    >>> viz = TicketVisualizer(output_dir="reports/figures")
    >>> viz.plot_category_distribution(df)
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

logger = logging.getLogger(__name__)


class TicketVisualizer:
    """Visualization engine for support ticket analysis.

    Produces figures suitable for both exploratory analysis and
    final reports/presentations.

    Attributes:
        style: Matplotlib style name.
        palette: Color palette name.
        output_dir: Directory to save figures.
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        palette: str = "viridis",
        output_dir: str = "reports/figures",
    ) -> None:
        """Initialize visualizer.

        Args:
            style: Matplotlib style.
            palette: Seaborn color palette.
            output_dir: Figure output directory.
        """
        self.style = style
        self.palette = palette
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use(style)
        sns.set_palette(palette)
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["font.size"] = 10

    def plot_category_distribution(self, df: pd.DataFrame, column: str = "category") -> None:
        """Plot ticket distribution across categories.

        Args:
            df: DataFrame containing ticket data.
            column: Name of the category column.
        """
        plt.figure(figsize=(10, 6))

        counts = df[column].value_counts()
        colors = sns.color_palette(self.palette, len(counts))

        ax = sns.barplot(x=counts.index, y=counts.values, palette=colors)

        # Add value labels on bars
        for i, v in enumerate(counts.values):
            ax.text(
                i, v + max(counts.values) * 0.01, f"{v}\n({v/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontweight="bold", fontsize=9
            )

        plt.title("Ticket Distribution by Category", fontsize=14, fontweight="bold")
        plt.xlabel("Category", fontsize=12)
        plt.ylabel("Number of Tickets", fontsize=12)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        save_path = self.output_dir / "01_category_distribution.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("Category distribution saved to %s", save_path)

    def plot_category_pie(self, df: pd.DataFrame, column: str = "category") -> None:
        """Plot pie chart of category distribution.

        Args:
            df: DataFrame containing ticket data.
            column: Name of the category column.
        """
        plt.figure(figsize=(8, 8))
        counts = df[column].value_counts()
        colors = sns.color_palette(self.palette, len(counts))

        wedges, texts, autotexts = plt.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            explode=[0.02] * len(counts),
            shadow=True,
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight("bold")

        plt.title("Category Proportion", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = self.output_dir / "02_category_pie.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("Category pie chart saved to %s", save_path)

    def plot_priority_distribution(self, df: pd.DataFrame) -> None:
        """Plot priority distribution.

        Args:
            df: DataFrame with 'priority' column.
        """
        plt.figure(figsize=(8, 5))
        counts = df["priority"].value_counts()
        colors = sns.color_palette("rocket", len(counts))

        ax = sns.barplot(x=counts.index, y=counts.values, palette=colors, order=["Low", "Medium", "High", "Critical"])
        for i, v in enumerate(counts.reindex(["Low", "Medium", "High", "Critical"]).values):
            if not np.isnan(v):
                ax.text(i, v + max(counts.values) * 0.01, f"{int(v)}", ha="center", va="bottom", fontweight="bold")

        plt.title("Ticket Distribution by Priority", fontsize=14, fontweight="bold")
        plt.xlabel("Priority Level", fontsize=12)
        plt.ylabel("Number of Tickets", fontsize=12)
        plt.tight_layout()

        save_path = self.output_dir / "03_priority_distribution.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("Priority distribution saved to %s", save_path)

    def plot_priority_by_category(self, df: pd.DataFrame) -> None:
        """Plot heatmap of priority vs category.

        Args:
            df: DataFrame with 'category' and 'priority' columns.
        """
        crosstab = pd.crosstab(df["category"], df["priority"], normalize="index")

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            crosstab,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            cbar_kws={"label": "Proportion"},
            linewidths=0.5,
        )

        plt.title("Priority Distribution by Category (Normalized)", fontsize=14, fontweight="bold")
        plt.xlabel("Priority", fontsize=12)
        plt.ylabel("Category", fontsize=12)
        plt.tight_layout()

        save_path = self.output_dir / "04_priority_heatmap.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("Priority heatmap saved to %s", save_path)

    def plot_text_length_distribution(self, df: pd.DataFrame, text_col: str = "text") -> None:
        """Plot distribution of text lengths by category.

        Args:
            df: DataFrame with text data.
            text_col: Name of the text column.
        """
        df = df.copy()
        df["char_count"] = df[text_col].str.len()
        df["word_count"] = df[text_col].str.split().str.len()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Character length histogram
        sns.histplot(data=df, x="char_count", hue="category", kde=True, ax=axes[0, 0], palette=self.palette)
        axes[0, 0].set_title("Character Count Distribution", fontweight="bold")
        axes[0, 0].set_xlabel("Characters")

        # Word count histogram
        sns.histplot(data=df, x="word_count", hue="category", kde=True, ax=axes[0, 1], palette=self.palette)
        axes[0, 1].set_title("Word Count Distribution", fontweight="bold")
        axes[0, 1].set_xlabel("Words")

        # Character length boxplot
        sns.boxplot(data=df, x="category", y="char_count", ax=axes[1, 0], palette=self.palette)
        axes[1, 0].set_title("Character Length by Category", fontweight="bold")
        axes[1, 0].set_xlabel("Category")
        axes[1, 0].tick_params(axis="x", rotation=30)

        # Word count boxplot
        sns.boxplot(data=df, x="category", y="word_count", ax=axes[1, 1], palette=self.palette)
        axes[1, 1].set_title("Word Count by Category", fontweight="bold")
        axes[1, 1].set_xlabel("Category")
        axes[1, 1].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        save_path = self.output_dir / "05_text_length_distribution.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("Text length distribution saved to %s", save_path)

    def plot_wordcloud(self, texts: List[str], category: str) -> None:
        """Generate word cloud for a specific category.

        Args:
            texts: List of preprocessed texts from one category.
            category: Category name for title and filename.
        """
        text = " ".join(texts)

        wordcloud = WordCloud(
            width=1200,
            height=800,
            background_color="white",
            colormap="viridis",
            max_words=200,
            relative_scaling=0.5,
            contour_width=1,
            contour_color="steelblue",
        ).generate(text)

        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud: {category}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        safe_name = category.lower().replace(" ", "_")
        save_path = self.output_dir / f"06_wordcloud_{safe_name}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("Word cloud for %s saved", category)

    def plot_temporal_trends(self, df: pd.DataFrame) -> None:
        """Plot ticket volume over time.

        Args:
            df: DataFrame with 'created_at' column.
        """
        df = df.copy()
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["date"] = df["created_at"].dt.date

        daily_counts = df.groupby(["date", "category"]).size().unstack(fill_value=0)

        plt.figure(figsize=(14, 7))
        daily_counts.plot(kind="area", stacked=True, alpha=0.7, colormap="viridis")
        plt.title("Daily Ticket Volume by Category", fontsize=14, fontweight="bold")
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Number of Tickets", fontsize=12)
        plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        save_path = self.output_dir / "07_temporal_trends.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("Temporal trends saved to %s", save_path)

    def plot_class_imbalance(self, df: pd.DataFrame) -> None:
        """Visualize class imbalance with imbalance ratio.

        Args:
            df: DataFrame with 'category' column.
        """
        counts = df["category"].value_counts()
        max_count = counts.max()
        imbalance_ratios = max_count / counts

        fig, ax1 = plt.subplots(figsize=(10, 6))

        colors = sns.color_palette(self.palette, len(counts))
        bars = ax1.bar(counts.index, counts.values, color=colors, alpha=0.8)
        ax1.set_ylabel("Number of Tickets", fontsize=12)
        ax1.set_title("Class Imbalance Analysis", fontsize=14, fontweight="bold")

        # Add count labels
        for bar, count in zip(bars, counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{count}", ha="center", va="bottom", fontweight="bold")

        # Imbalance ratio line
        ax2 = ax1.twinx()
        ax2.plot(counts.index, imbalance_ratios.values, "ro-", linewidth=2, markersize=8, label="Imbalance Ratio")
        ax2.set_ylabel("Imbalance Ratio (majority / class)", fontsize=12, color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.axhline(y=1.5, color="orange", linestyle="--", alpha=0.7, label="Moderate Imbalance (1.5x)")
        ax2.axhline(y=2.0, color="red", linestyle="--", alpha=0.7, label="Severe Imbalance (2.0x)")
        ax2.legend(loc="upper right")

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        save_path = self.output_dir / "08_class_imbalance.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("Class imbalance plot saved to %s", save_path)

    def plot_ngrams(self, df: pd.DataFrame, category: str, n: int = 2, top_k: int = 15) -> None:
        """Plot top N-grams for a specific category.

        Args:
            df: DataFrame with 'text' and 'category' columns.
            category: Category to analyze.
            n: N-gram size (1 for unigram, 2 for bigram).
            top_k: Number of top N-grams to show.
        """
        from collections import Counter
        import re

        texts = df[df["category"] == category]["text"].tolist()
        all_text = " ".join(texts).lower()
        all_text = re.sub(r"[^\w\s]", "", all_text)
        tokens = all_text.split()

        if n == 1:
            ngrams = tokens
            title = f"Top Unigrams: {category}"
            fname = f"09_unigrams_{category.lower().replace(' ', '_')}.png"
        else:
            ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            title = f"Top Bigrams: {category}"
            fname = f"09_bigrams_{category.lower().replace(' ', '_')}.png"

        # Remove common stopwords for unigrams
        if n == 1:
            stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
                        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
                        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
                        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
                        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
                        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
                        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
                        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
                        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "can",
                        "will", "just", "should", "now", "please", "help", "need", "want", "like", "get"}
            ngrams = [ng for ng in ngrams if ng not in stopwords and len(ng) > 2]

        counter = Counter(ngrams)
        top_ngrams = counter.most_common(top_k)
        labels, values = zip(*top_ngrams) if top_ngrams else ([], [])

        plt.figure(figsize=(10, 6))
        colors = sns.color_palette(self.palette, len(labels))
        bars = plt.barh(range(len(labels)), values, color=colors)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Frequency", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()

        for bar, val in zip(bars, values):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / fname
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info("%s saved to %s", title, save_path)
