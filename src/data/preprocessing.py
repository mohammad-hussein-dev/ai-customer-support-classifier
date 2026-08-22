"""
Text Preprocessing Pipeline for Banking Support Tickets
=======================================================
Cleans, tokenizes, and normalizes customer service messages.
Optimized for banking vocabulary and short-form text.

Author: Mohammad Hussein
"""

import logging
import re
import string
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

BANKING_STOPWORDS = {
    "please", "help", "need", "want", "like", "get", "would", "could",
    "should", "just", "also", "really", "still", "even", "much", "many",
    "thing", "things", "way", "ways", "time", "times", "day", "days",
    "week", "weeks", "month", "months", "year", "years",
}

_NLTK_RESOURCES = ["punkt", "punkt_tab", "stopwords", "wordnet"]
for _res in _NLTK_RESOURCES:
    try:
        nltk.data.find(f"tokenizers/{_res}" if _res in ("punkt", "punkt_tab") else f"corpora/{_res}")
    except LookupError:
        print(f"[*] Downloading NLTK resource: {_res} ...")
        nltk.download(_res, quiet=True)
        print(f"[+] {_res} downloaded.")


class TextPreprocessor:
    """Production-grade text preprocessor for banking support tickets.
    
    Pipeline:
        1. Lowercase
        2. Remove URLs, emails, mentions
        3. Remove numbers (optional)
        4. Remove punctuation
        5. Tokenize
        6. Filter by length
        7. Remove stopwords (general + banking-specific)
        8. Lemmatize
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
        custom_stopwords: Optional[List[str]] = None,
    ) -> None:
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length

        self.lemmatizer = WordNetLemmatizer() if lemmatize else None

        self.stop_words = set()
        if remove_stopwords:
            self.stop_words = set(stopwords.words("english"))
            self.stop_words.update(BANKING_STOPWORDS)
            if custom_stopwords:
                self.stop_words.update(custom_stopwords)

        logger.info(
            "TextPreprocessor initialized | lowercase=%s, punct=%s, stopwords=%s, lemma=%s",
            lowercase, remove_punctuation, remove_stopwords, lemmatize,
        )

    def clean(self, text: str) -> str:
        """Apply full preprocessing pipeline to a single text."""
        if not isinstance(text, str):
            logger.warning("Non-string input: %s", type(text))
            return ""

        if self.lowercase:
            text = text.lower()

        text = re.sub(r"http\S+|www\S+|@\w+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        
        if self.remove_numbers:
            text = re.sub(r"\b\d+\b", "", text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        tokens = word_tokenize(text)

        filtered = []
        for token in tokens:
            if not (self.min_word_length <= len(token) <= self.max_word_length):
                continue
            if self.remove_stopwords and token in self.stop_words:
                continue
            if self.lemmatize and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            filtered.append(token)

        return " ".join(filtered)

    def transform(self, texts: List[str]) -> List[str]:
        """Apply preprocessing to a list of texts."""
        logger.info("Preprocessing %d documents", len(texts))
        return [self.clean(t) for t in texts]

    def fit_transform(self, texts: List[str]) -> List[str]:
        """Fit (no-op) and transform texts."""
        return self.transform(texts)
