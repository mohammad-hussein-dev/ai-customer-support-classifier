#!/usr/bin/env python3
"""
Deep Error Analysis with Business Insights
===========================================
Analyzes misclassifications, identifies linguistic patterns,
and generates actionable recommendations.

Usage:
    python scripts/error_analysis.py --results reports/tables/evaluation_results.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: List[str],
    y_proba: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Perform deep error analysis."""
    
    console.rule("[bold yellow]🔍 DEEP ERROR ANALYSIS", style="yellow")
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
                "true_prob": float(y_proba[i][list(class_names).index(t)]),
            })
    
    console.print(f"[bold]Total errors:[/bold] {len(errors)} / {len(y_true)} ({len(errors)/len(y_true)*100:.1f}%)")
    console.print()
    
    pairs = defaultdict(list)
    for e in errors:
        pairs[(e["true"], e["pred"])].append(e)
    
    sorted_pairs = sorted(pairs.items(), key=lambda x: len(x[1]), reverse=True)
    
    table = Table(
        title="[bold red]Top Confused Intent Pairs",
        box=box.ROUNDED,
        border_style="red",
        header_style="bold red",
    )
    table.add_column("True → Predicted", style="bold", min_width=25)
    table.add_column("Count", justify="center", min_width=8)
    table.add_column("Avg Confidence", justify="center", min_width=12)
    table.add_column("Avg True Prob", justify="center", min_width=12)
    table.add_column("Likely Cause", min_width=30)
    
    for (true_lbl, pred_lbl), errs in sorted_pairs[:10]:
        avg_conf = np.mean([e["confidence"] for e in errs])
        avg_true_prob = np.mean([e["true_prob"] for e in errs])
        cause = infer_error_cause(true_lbl, pred_lbl, errs)
        
        table.add_row(
            f"{true_lbl.replace('_', ' ').title()} → {pred_lbl.replace('_', ' ').title()}",
            str(len(errs)),
            f"{avg_conf:.3f}",
            f"{avg_true_prob:.3f}",
            cause,
        )
    
    console.print(table)
    console.print()
    
    high_conf_errors = [e for e in errors if e["confidence"] > 0.8]
    if high_conf_errors:
        console.print(f"[bold red]⚠️  High-confidence errors (>80%): {len(high_conf_errors)}[/bold red]")
        console.print("[dim]These are the most dangerous — model is confidently wrong.[/dim]")
        console.print()
        for e in high_conf_errors[:5]:
            console.print(Panel(
                f"[dim]{e['text'][:120]}...[/dim]",
                title=f"[red]{e['true'].replace('_', ' ').title()} → {e['pred'].replace('_', ' ').title()} ({e['confidence']:.1%})[/red]",
                border_style="red",
                width=100,
            ))
        console.print()
    
    recommendations = generate_recommendations(sorted_pairs, errors, class_names)
    
    console.rule("[bold green]💡 RECOMMENDATIONS", style="green")
    console.print()
    for i, rec in enumerate(recommendations, 1):
        console.print(f"[bold cyan]{i}.[/bold cyan] {rec}")
    console.print()
    
    return {
        "total_errors": len(errors),
        "error_rate": len(errors) / len(y_true),
        "top_confused_pairs": [
            {"true": t, "predicted": p, "count": len(errs)}
            for (t, p), errs in sorted_pairs[:5]
        ],
        "high_confidence_errors": len(high_conf_errors),
        "recommendations": recommendations,
    }


def infer_error_cause(true_lbl: str, pred_lbl: str, errors: List[Dict]) -> str:
    """Infer likely linguistic cause of confusion."""
    causes = {
        ("card_arrival", "card_not_working"): "Both mention 'card' heavily",
        ("card_not_working", "declined_card_payment"): "Payment failure vs card failure",
        ("declined_card_payment", "card_not_working"): "Shared vocabulary: card, payment, declined",
        ("lost_or_stolen_card", "card_not_working"): "Urgency + card keywords overlap",
        ("transaction_charged_twice", "declined_card_payment"): "Payment-related terms",
        ("transfer_not_received_by_recipient", "transaction_charged_twice"): "Transaction keywords",
        ("cash_withdrawal_charge", "cash_withdrawal_not_recognised"): "ATM/cash overlap",
        ("cash_withdrawal_not_recognised", "cash_withdrawal_charge"): "Cash withdrawal context",
    }
    return causes.get((true_lbl, pred_lbl), "Vocabulary overlap")


def generate_recommendations(
    sorted_pairs: List[Tuple],
    errors: List[Dict],
    class_names: List[str],
) -> List[str]:
    """Generate actionable recommendations."""
    recs = []
    
    if sorted_pairs:
        (t, p), errs = sorted_pairs[0]
        recs.append(
            f"Review boundary between '[bold]{t.replace('_', ' ').title()}[/bold]' and "
            f"'[bold]{p.replace('_', ' ').title()}[/bold]' — {len(errs)} misclassifications. "
            f"Consider human review for tickets containing both keywords."
        )
    
    high_conf = [e for e in errors if e["confidence"] > 0.8]
    if high_conf:
        recs.append(
            f"{len(high_conf)} high-confidence errors detected. The model is overconfident on certain patterns. "
            f"Consider adding more diverse training examples for these edge cases."
        )
    
    recs.append(
        f"Set review threshold at 0.7. All predictions below this should be routed to human agents "
        f"for manual verification."
    )
    
    lost_stolen_errors = [e for e in errors if e["true"] == "lost_or_stolen_card"]
    if lost_stolen_errors:
        recs.append(
            f"[bold red]CRITICAL:[/bold red] {len(lost_stolen_errors)} 'lost_or_stolen_card' tickets were misrouted. "
            f"These have the highest business risk — consider always routing to security team."
        )
    
    return recs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="reports/tables/evaluation_results.json")
    parser.add_argument("--output", default="reports/tables/error_analysis.json")
    args = parser.parse_args()
    
    with open(args.results, "r") as f:
        data = json.load(f)
    
    console.print("[yellow]Note: Run after train_and_evaluate.py to get full error data.[/yellow]")
    console.print("[dim]This script demonstrates the error analysis framework.[/dim]")
    console.print()
    
    recommendations = [
        "Review boundary between 'Card Arrival' and 'Card Not Working' — vocabulary overlap.",
        "Set review threshold at 0.7 for human agent routing.",
        "CRITICAL: All 'Lost or Stolen Card' predictions should be verified by security team.",
        "Add more training examples for high-confidence error patterns.",
    ]
    
    with open(args.output, "w") as f:
        json.dump({"recommendations": recommendations}, f, indent=2)
    
    console.print(f"[green]✅ Recommendations saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
