"""
Rich Terminal Metrics & Utilities
=================================
Beautiful console output for model training and evaluation.

Author: Mohammad Hussein
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich import box

console = Console()


class MetricsPrinter:
    """Print beautiful metric tables to the terminal."""

    @staticmethod
    def print_header(title: str, subtitle: str = "") -> None:
        """Print a styled header."""
        console.print()
        console.rule(f"[bold cyan]{title}", style="cyan")
        if subtitle:
            console.print(f"[dim]{subtitle}[/dim]", justify="center")
        console.print()

    @staticmethod
    def print_config(config: Dict[str, Any], title: str = "Configuration") -> None:
        """Print configuration as a styled table."""
        table = Table(title=f"[bold]{title}", box=box.ROUNDED, border_style="blue", width=60)
        table.add_column("Parameter", style="bold cyan", min_width=20)
        table.add_column("Value", style="white", min_width=20)
        
        for key, value in config.items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            table.add_row(str(key), str(value))
        
        console.print(table)
        console.print()

    @staticmethod
    def print_cv_results(cv_results: Dict[str, List[float]], model_name: str) -> None:
        """Print cross-validation results in a beautiful table."""
        table = Table(
            title=f"[bold cyan]📊 Cross-Validation Results: {model_name}",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold", min_width=15)
        table.add_column("Mean", justify="center", min_width=12)
        table.add_column("Std", justify="center", min_width=12)
        table.add_column("Min", justify="center", min_width=12)
        table.add_column("Max", justify="center", min_width=12)
        table.add_column("Fold Scores", min_width=30)

        for metric, scores in cv_results.items():
            if metric.startswith("test_"):
                clean_name = metric.replace("test_", "").replace("_", " ").title()
                scores_arr = np.array(scores)
                fold_str = " ".join([f"{s:.3f}" for s in scores_arr])
                
                mean_val = np.mean(scores_arr)
                color = "green" if mean_val >= 0.85 else "yellow" if mean_val >= 0.70 else "red"
                
                table.add_row(
                    clean_name,
                    f"[bold {color}]{mean_val:.4f}[/bold {color}]",
                    f"{np.std(scores_arr):.4f}",
                    f"{np.min(scores_arr):.4f}",
                    f"{np.max(scores_arr):.4f}",
                    f"[dim]{fold_str}[/dim]",
                )
        
        console.print(table)
        console.print()

    @staticmethod
    def print_model_comparison(results: List[Dict[str, Any]]) -> None:
        """Print model comparison table."""
        table = Table(
            title="[bold cyan]🏆 Model Comparison",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
        )
        table.add_column("Rank", justify="center", min_width=6)
        table.add_column("Model", style="bold", min_width=20)
        table.add_column("CV F1", justify="center", min_width=10)
        table.add_column("Test F1", justify="center", min_width=10)
        table.add_column("Accuracy", justify="center", min_width=10)
        table.add_column("Size", justify="center", min_width=10)
        table.add_column("Status", justify="center", min_width=12)

        # Sort by test F1
        sorted_results = sorted(results, key=lambda x: x.get("test_f1", 0), reverse=True)
        
        medals = ["🥇", "🥈", "🥉"]
        for idx, res in enumerate(sorted_results):
            rank = medals[idx] if idx < 3 else f"{idx+1}."
            test_f1 = res.get("test_f1", 0)
            status = "[bold green]✅ Best[/bold green]" if idx == 0 else "[dim]—[/dim]"
            
            color = "green" if test_f1 >= 0.85 else "yellow" if test_f1 >= 0.70 else "red"
            
            table.add_row(
                rank,
                res["model_name"],
                f"{res.get('cv_f1', 0):.4f}",
                f"[bold {color}]{test_f1:.4f}[/bold {color}]",
                f"{res.get('accuracy', 0):.4f}",
                res.get("model_size", "—"),
                status,
            )
        
        console.print(table)
        console.print()


@contextmanager
def timer(name: str = "Operation"):
    """Context manager to time operations with Rich output."""
    start = time.time()
    console.print(f"[dim]⏳ {name} started...[/dim]")
    try:
        yield
    finally:
        elapsed = time.time() - start
        console.print(f"[dim]✅ {name} completed in {elapsed:.2f}s[/dim]")
        console.print()


def print_pipeline_stage(stage_num: int, total: int, title: str, description: str = "") -> None:
    """Print a pipeline stage header."""
    console.print()
    console.rule(f"[bold yellow]Stage {stage_num}/{total}: {title}", style="yellow")
    if description:
        console.print(f"[dim]{description}[/dim]")
    console.print()


def print_file_saved(path: str, file_type: str = "file") -> None:
    """Print file save confirmation."""
    console.print(f"[green]💾 {file_type.capitalize()} saved:[/green] [cyan]{path}[/cyan]")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]❌ {message}[/bold red]")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✅ {message}[/bold green]")
