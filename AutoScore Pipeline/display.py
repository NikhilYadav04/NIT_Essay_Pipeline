"""
display.py — Visual terminal output for AutoSCORE results
Rich terminal UI with 3 sections:
  1. Evidence JSON panel (Agent 1 output)
  2. Per-trait score table + detail panels (Agent 2 output)
  3. Final holistic score + audit trail
"""

import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.rule import Rule
from rich import box

console = Console()

TRAIT_LABELS = [
    "Task Response",
    "Argument Quality",
    "Organisation",
    "Language & Style",
    "Grammar & Mechanics",
]
TRAIT_KEYS = [
    "trait_1_task_response",
    "trait_2_argument_quality",
    "trait_3_organisation",
    "trait_4_language_style",
    "trait_5_grammar_mechanics",
]
TRAIT_COLORS = ["cyan", "blue", "green", "yellow", "magenta"]
TRAIT_ICONS  = ["📝", "🧠", "📐", "📚", "✏️ "]


# ── Helpers ────────────────────────────────────────────────────────────────────

def score_color(score: int) -> str:
    if score <= 2:
        return "bold red"
    elif score <= 4:
        return "bold yellow"
    else:
        return "bold green"


def score_bar(score: int, max_score: int = 6) -> str:
    filled = round((score / max_score) * 12)
    empty  = 12 - filled
    color  = score_color(score)
    bar    = "█" * filled + "░" * empty
    return f"[{color}]{bar}  {score}/6[/{color}]"


def truncate(text: str, max_len: int = 55) -> str:
    return (text[:max_len] + "...") if len(text) > max_len else text


# ── Main display ───────────────────────────────────────────────────────────────

def display_results(
    results: dict,
    backend: str = "gemini",
    model: str = "gemini-2.0-flash",
    example: str = "",
):
    backend_color = "cyan" if backend == "ollama" else "blue"
    example_label = f"  ·  [dim]essay: {example}[/dim]" if example else ""

    console.print()
    console.rule("[bold blue] 🔬  AutoSCORE — Two-Agent Essay Evaluation [/bold blue]")
    console.print(
        f"  [{backend_color}]🔧 Backend:[/{backend_color}] [bold]{backend.upper()}[/bold]"
        f"   [{backend_color}]⚙  Model:[/{backend_color}] [bold]{model}[/bold]"
        + example_label
    )
    console.print()

    # ── Section 1: Evidence JSON (Agent 1 output) ─────────────────────────────
    console.rule("[bold cyan] 📋  Step 1 — Agent 1 (SRCE): Structured Evidence [/bold cyan]")
    console.print(
        "[dim]  Agent 1 read the rubric and essay. "
        "It extracted evidence only — no scores assigned.[/dim]\n"
    )

    evidence_str = results.get("evidence_json", "{}")
    if not evidence_str or evidence_str == "{}":
        evidence_str = json.dumps(results.get("evidence_dict", {}), indent=2)

    syntax = Syntax(
        evidence_str,
        "json",
        theme="monokai",
        line_numbers=False,
        word_wrap=True,
    )
    console.print(Panel(
        syntax,
        title="[bold cyan]Evidence JSON — Agent 1 Output[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    # ── Section 2: Per-trait scores (Agent 2 output) ──────────────────────────
    console.rule("[bold blue] ⚖️   Step 2 — Agent 2 (Scoring): Rubric Judgment Applied [/bold blue]")
    console.print(
        "[dim]  Agent 2 used the evidence JSON as its primary source. "
        "The essay was provided as reference only.[/dim]\n"
    )

    scoring      = results.get("scoring_dict", {})
    trait_scores = scoring.get("trait_scores", {})

    # Score summary table
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="dim blue",
        padding=(0, 1),
    )
    table.add_column("Trait",         style="bold", width=22)
    table.add_column("Score",                       width=22)
    table.add_column("Evidence Used",               width=50)

    for key, label, color, icon in zip(TRAIT_KEYS, TRAIT_LABELS, TRAIT_COLORS, TRAIT_ICONS):
        trait         = trait_scores.get(key, {})
        score         = trait.get("score", 0)
        evidence_used = truncate(trait.get("evidence_used", "—"))
        table.add_row(
            f"[{color}]{icon} {label}[/{color}]",
            score_bar(score),
            f"[dim]{evidence_used}[/dim]",
        )

    console.print(table)
    console.print()

    # Per-trait detail panels
    console.print("[bold white]◆  Per-Trait Reasoning and Feedback[/bold white]\n")

    for key, label, color, icon in zip(TRAIT_KEYS, TRAIT_LABELS, TRAIT_COLORS, TRAIT_ICONS):
        trait         = trait_scores.get(key, {})
        score         = trait.get("score", 0)
        rubric_match  = trait.get("rubric_match", "—")
        evidence_used = trait.get("evidence_used", "—")
        feedback      = trait.get("feedback", "—")

        content = (
            f"[bold]Rubric Match:[/bold]   {rubric_match}\n\n"
            f"[bold]Evidence Used:[/bold]  {evidence_used}\n\n"
            f"[bold]Student Feedback:[/bold]\n{feedback}"
        )
        console.print(Panel(
            content,
            title=f"{icon} [bold {color}]{label}[/bold {color}]   {score_bar(score)}",
            border_style=color,
            padding=(1, 2),
            expand=True,
        ))
        console.print()

    # ── Section 3: Holistic score ─────────────────────────────────────────────
    console.rule("[bold green] 🎯  Final Holistic Score [/bold green]")
    console.print()

    holistic_score   = scoring.get("holistic_score", 0)
    holistic_reason  = scoring.get("holistic_reasoning", "—")
    overall_feedback = scoring.get("overall_feedback", "—")
    s_color          = score_color(holistic_score)

    console.print(Panel(
        f"[{s_color}]HOLISTIC SCORE:  {holistic_score} / 6[/{s_color}]\n\n"
        f"{score_bar(holistic_score)}\n\n"
        f"[bold]Reasoning:[/bold]\n{holistic_reason}\n\n"
        f"[bold]Overall Student Feedback:[/bold]\n{overall_feedback}",
        title="[bold green]🎯  Final Evaluation[/bold green]",
        border_style="green",
        padding=(1, 3),
        expand=True,
    ))

    # ── Audit trail footer ────────────────────────────────────────────────────
    console.print()
    console.rule("[dim]Audit Trail[/dim]")
    console.print(
        "[dim]  Agent 1 extracted evidence  →  "
        "Agent 2 scored from evidence  →  "
        "Score above[/dim]"
    )
    console.print(
        "[dim]  Every score is traceable to a specific field in the Evidence JSON above.[/dim]"
    )
    console.rule("[dim]End of AutoSCORE Evaluation[/dim]")
    console.print()
