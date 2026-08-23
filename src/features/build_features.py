"""Feature engineering pipeline for ticket classification.

Builds TF-IDF and additional text-based features for machine learning models.
"""

import logging
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Constructs feature matrices from preprocessed text data.

    Combines TF-IDF vectorization with optional engineered features
    such as text length, word count, and sentiment proxies.

    Attributes:
        vectorizer: Fitted TF-IDF vectorizer instance.
        include_meta: Whether to include metadata features.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        include_meta: bool = False,
    ) -> None:
        """Initialize feature builder.

        Args:
            max_features: Maximum number of TF-IDF features.
            ngram_range: N-gram range for tokenization.
            min_df: Minimum document frequency for terms.
            max_df: Maximum document frequency for terms.
            sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
            include_meta: Whether to add metadata features.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.include_meta = include_meta

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            dtype=np.float32,
        )

        logger.info(
            "FeatureBuilder initialized: max_features=%d, ngram_range=%s",
            max_features,
            ngram_range,
        )

    def fit(self, texts: List[str]) -> "FeatureBuilder":
        """Fit the vectorizer on training data.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            Self for method chaining.
        """
        n_documents = len(texts)

        if n_documents == 0:
            raise ValueError("Cannot fit FeatureBuilder on empty text collection.")

        # Small datasets need relaxed document-frequency thresholds.
        # Otherwise sklearn can prune every term before fitting.
        if n_documents < 3:
            effective_min_df = 1
            effective_max_df = 1.0
        else:
            effective_min_df = self.min_df
            effective_max_df = self.max_df

            if isinstance(self.max_df, float) and 0 < self.max_df < 1:
                max_df_documents = int(self.max_df * n_documents)

                if max_df_documents < effective_min_df:
                    effective_max_df = 1.0

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=effective_min_df,
            max_df=effective_max_df,
            sublinear_tf=self.sublinear_tf,
            strip_accents="unicode",
            dtype=np.float32,
        )

        logger.info("Fitting vectorizer on %d documents", n_documents)
        self.vectorizer.fit(texts)
        logger.info("Vocabulary size: %d", len(self.vectorizer.vocabulary_))
        return self

    def transform(self, texts: List[str]) -> csr_matrix:
        """Transform texts to feature matrix.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            Sparse feature matrix.
        """
        tfidf_matrix = self.vectorizer.transform(texts)

        if self.include_meta:
            meta = self._extract_meta_features(texts)
            tfidf_matrix = hstack([tfidf_matrix, meta], format="csr")

        return tfidf_matrix

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Fit and transform in one step.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            Sparse feature matrix.
        """
        return self.fit(texts).transform(texts)

    def _extract_meta_features(self, texts: List[str]) -> csr_matrix:
        """Extract metadata features from texts.

        Features:
            - Character count
            - Word count
            - Average word length
            - Exclamation mark count (urgency proxy)
            - Question mark count

        Args:
            texts: List of text strings.

        Returns:
            Sparse matrix of metadata features.
        """
        features = []
        for text in texts:
            words = text.split()
            char_count = len(text)
            word_count = len(words)
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            exclamation_count = text.count("!")
            question_count = text.count("?")

            features.append(
                [
                    char_count,
                    word_count,
                    avg_word_len,
                    exclamation_count,
                    question_count,
                ]
            )

        return csr_matrix(np.array(features))

    def get_feature_names(self) -> List[str]:
        """Return feature names from the vectorizer.

        Returns:
            List of feature names.
        """
        return list(self.vectorizer.get_feature_names_out())
