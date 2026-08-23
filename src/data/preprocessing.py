"""
Text preprocessing pipeline for banking support tickets.

This module provides a configurable text preprocessing pipeline designed
for short-form customer support messages in the banking domain.

Pipeline:
    1. Normalize letter casing
    2. Remove URLs, email addresses, and mentions
    3. Optionally remove standalone numbers
    4. Remove punctuation
    5. Tokenize text
    6. Filter tokens by length
    7. Remove general and banking-specific stopwords
    8. Optionally lemmatize tokens

Author:
    Mohammad Hussein
"""

import logging
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


# Domain-specific words that usually carry little predictive value
# for banking customer-support intent classification.
BANKING_STOPWORDS = {
    "please",
    "help",
    "need",
    "want",
    "like",
    "get",
    "would",
    "could",
    "should",
    "just",
    "also",
    "really",
    "still",
    "even",
    "much",
    "many",
    "thing",
    "things",
    "way",
    "ways",
    "time",
    "times",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
}


# NLTK resources required by this preprocessing pipeline.
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet.zip",
}


def _ensure_nltk_resources() -> None:
    """Ensure all required NLTK resources are available locally.

    Missing resources are downloaded automatically. This keeps the
    preprocessing pipeline usable on a fresh development environment.
    """

    for resource_name, resource_path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading missing NLTK resource: %s", resource_name)

            nltk.download(resource_name, quiet=True)

            # Verify that the resource is now available.
            try:
                nltk.data.find(resource_path)
            except LookupError as exc:
                raise RuntimeError(
                    f"Failed to download required NLTK resource: {resource_name}"
                ) from exc


_ensure_nltk_resources()


class TextPreprocessor:
    """Production-oriented text preprocessor for banking support tickets.

    The pipeline is intentionally configurable so the same component can
    be reused during training, evaluation, and inference.

    Attributes:
        lowercase: Convert text to lowercase.
        remove_punctuation: Remove punctuation characters.
        remove_stopwords: Remove general and banking-specific stopwords.
        lemmatize: Reduce words to their base form.
        remove_numbers: Remove standalone numeric tokens.
        min_word_length: Minimum accepted token length.
        max_word_length: Maximum accepted token length.
        stop_words: Combined stopword set used during preprocessing.
        lemmatizer: NLTK WordNet lemmatizer when enabled.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        remove_numbers: bool = True,
        min_word_length: int = 2,
        max_word_length: int = 20,
        custom_stopwords: list[str] | None = None,
    ) -> None:
        """Initialize the text preprocessing pipeline.

        Args:
            lowercase: Convert input text to lowercase.
            remove_punctuation: Remove punctuation characters.
            remove_stopwords: Remove general and domain-specific stopwords.
            lemmatize: Apply WordNet lemmatization.
            remove_numbers: Remove standalone numbers.
            min_word_length: Minimum token length to keep.
            max_word_length: Maximum token length to keep.
            custom_stopwords: Optional additional stopwords.

        Raises:
            ValueError: If token-length boundaries are invalid.
        """

        if min_word_length < 1:
            raise ValueError("min_word_length must be greater than or equal to 1.")

        if max_word_length < min_word_length:
            raise ValueError(
                "max_word_length must be greater than or equal to " "min_word_length."
            )

        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length

        self.lemmatizer = WordNetLemmatizer() if lemmatize else None

        self.stop_words: set[str] = set()

        if remove_stopwords:
            self.stop_words.update(stopwords.words("english"))
            self.stop_words.update(BANKING_STOPWORDS)

            if custom_stopwords:
                self.stop_words.update(custom_stopwords)

        logger.info(
            "TextPreprocessor initialized | "
            "lowercase=%s, punctuation=%s, stopwords=%s, "
            "lemmatize=%s, remove_numbers=%s",
            lowercase,
            remove_punctuation,
            remove_stopwords,
            lemmatize,
            remove_numbers,
        )

    def clean(self, text: str) -> str:
        """Clean and normalize a single text sample.

        Args:
            text: Raw customer-support message.

        Returns:
            A normalized string containing the retained tokens.

        Notes:
            Non-string input is treated as invalid and converted to an
            empty string instead of raising an exception. This makes the
            preprocessing stage more robust against imperfect datasets.
        """

        if not isinstance(text, str):
            logger.warning("Expected string input, received %s.", type(text))
            return ""

        if not text.strip():
            return ""

        # Normalize casing before applying the remaining transformations.
        if self.lowercase:
            text = text.lower()

        # Remove URLs, mentions, and email addresses.
        text = re.sub(r"http\S+|www\S+|@\w+", "", text)
        text = re.sub(r"\S+@\S+", "", text)

        # Remove standalone numeric values while preserving numbers
        # embedded inside words.
        if self.remove_numbers:
            text = re.sub(r"\b\d+\b", "", text)

        # Remove standard ASCII punctuation.
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Convert the normalized text into individual tokens.
        tokens = word_tokenize(text)

        filtered_tokens: list[str] = []

        for token in tokens:
            # Ignore tokens outside the configured length range.
            if not (self.min_word_length <= len(token) <= self.max_word_length):
                continue

            # Remove general and domain-specific stopwords.
            if self.remove_stopwords and token in self.stop_words:
                continue

            # Reduce words to their dictionary/base form.
            if self.lemmatize and self.lemmatizer is not None:
                token = self.lemmatizer.lemmatize(token)

            filtered_tokens.append(token)

        return " ".join(filtered_tokens)

    def transform(self, texts: list[str]) -> list[str]:
        """Preprocess a collection of text samples.

        Args:
            texts: Input customer-support messages.

        Returns:
            Preprocessed messages in the same order as the input.
        """

        logger.info("Preprocessing %d documents.", len(texts))

        return [self.clean(text) for text in texts]

    def fit_transform(self, texts: list[str]) -> list[str]:
        """Fit and transform the input texts.

        This preprocessor is stateless with respect to the input data,
        so fitting is currently a no-op. The method is provided to keep
        the component compatible with common ML preprocessing APIs.

        Args:
            texts: Input customer-support messages.

        Returns:
            Preprocessed messages.
        """

        return self.transform(texts)
