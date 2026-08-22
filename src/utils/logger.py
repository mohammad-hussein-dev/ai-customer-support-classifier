"""
Structured Logging with Rich Console Handler
=============================================
Beautiful, color-coded logs for terminal output.

Author: Mohammad Hussein
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "banking_classifier",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup a Rich-enabled logger with optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        show_time=True,
        console=Console(stderr=True),
    )
    console_handler.setLevel(level)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger
