"""
Utility Helpers for Banking Intent Classification
=================================================
Common utilities used across the pipeline.

Author: Mohammad Hussein
"""

import re
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(config_path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary as JSON."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def clean_text_basic(text: str) -> str:
    """Basic text cleaning without NLTK dependencies."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|\S+@\S+", "", text)
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + suffix


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_intent_name(intent: str) -> str:
    """Format intent name for display (snake_case → Title Case)."""
    return intent.replace("_", " ").title()
