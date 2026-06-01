"""
display.py — Visual terminal output for MAGIC pipeline results
Uses the `rich` library for a premium terminal UI.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import box

console = Console()

# ── Styling constants ──────────────────────────────────────────────────────────
ASPECT_COLORS = ["cyan", "blue", "magenta", "yellow", "green"]
ASPECT_ICONS  = ["📝", "🧠", "📐", "📚", "✏️ "]

AGENT_LABELS = [
    "T1 · Prompt Response",
    "T2 · Argumentation",
    "T3 · Organisation",
    "T4 · Vocabulary",
    "T5 · Grammar",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def score_color(score: int) -> str:
    if score <= 2:
        return "bold red"
    elif score <= 4:
        return "bold yellow"
    else:
        return "bold green"


def score_bar(score: int, max_score: int = 6) -> str:
    """Renders a Unicode block bar, e.g. '████████░░ 4/6'"""
    filled = round((score / max_score) * 10)
    empty  = 10 - filled
    color  = score_color(score)
    bar    = "█" * filled + "░" * empty
    return f"[{color}]{bar} {score}/{max_score}[/{color}]"


# ── Main display function ──────────────────────────────────────────────────────

def display_results(results: dict, backend: str = "gemini", model: str = "gemini-2.0-flash", example: str = ""):
    """
    Renders the full MAGIC pipeline output to the terminal.

    Expected keys in results:
        - agent_results:      list of AgentResult dicts (sorted by agent_index)
        - domain_scores:      list of 5 ints
        - orchestrator_score: int
        - final_feedback:     str
    """
    backend_color = "cyan" if backend == "ollama" else "blue"
    example_label = f" · [dim]essay: {example}[/dim]" if example else ""

    console.print()
    console.rule("[bold blue] ✨  MAGIC Essay Evaluation Pipeline  ✨ [/bold blue]")
    console.print(
        f"  [{backend_color}]🔧 Backend:[/{backend_color}] [bold]{backend.upper()}[/bold]"
        f"   [{backend_color}]⚙  Model:[/{backend_color}] [bold]{model}[/bold]"
        + example_label
    )
    console.print()
    console.print()

    # ── 1. Agent Scores Summary Table ─────────────────────────────────────────
    console.print("[bold white]◆  Step 1–5 · Specialised Agent Scores[/bold white]\n")

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="dim blue",
        padding=(0, 1),
    )
    table.add_column("#",       style="dim",   width=4)
    table.add_column("Agent",   style="bold",  width=14)
    table.add_column("Aspect",                 width=48)
    table.add_column("Score",                  width=20)

    for i, agent in enumerate(results["agent_results"]):
        color = ASPECT_COLORS[i]
        icon  = ASPECT_ICONS[i]
        table.add_row(
            f"[dim]{i + 1}[/dim]",
            f"[{color}]{icon} Agent {i + 1}[/{color}]",
            f"[{color}]{agent['aspect_name']}[/{color}]",
            score_bar(agent["score"]),
        )

    console.print(table)
    console.print()

    # ── 2. Per-agent Feedback Panels ──────────────────────────────────────────
    console.print("[bold white]◆  Detailed Reasoning per Agent[/bold white]\n")

    for i, agent in enumerate(results["agent_results"]):
        color = ASPECT_COLORS[i]
        icon  = ASPECT_ICONS[i]

        title = (
            f"{icon} [bold {color}]Agent {i + 1}[/bold {color}]"
            f" · {agent['aspect_name']}"
            f"   {score_bar(agent['score'])}"
        )

        comment = agent["examiner_comment"]

        console.print(Panel(
            comment,
            title=title,
            border_style=color,
            padding=(1, 2),
            expand=True,
        ))
        console.print()

    # ── 3. Score Summary Bar ──────────────────────────────────────────────────
    console.rule("[bold white]Score Summary[/bold white]")
    console.print()

    summary = Table(
        box=box.SIMPLE,
        show_header=False,
        padding=(0, 2),
        border_style="dim",
    )
    summary.add_column(width=32)
    summary.add_column(width=24)

    for i, score in enumerate(results["domain_scores"]):
        color = ASPECT_COLORS[i]
        icon  = ASPECT_ICONS[i]
        label = AGENT_LABELS[i]
        summary.add_row(
            f"[{color}]{icon} {label}[/{color}]",
            score_bar(score),
        )

    console.print(summary)
    console.print()

    # ── 4. Orchestrator Final Panel ───────────────────────────────────────────
    console.rule("[bold green] 🎯  Step 6 · Orchestrator Final Score [/bold green]")
    console.print()

    final_score = results["orchestrator_score"]
    s_color     = score_color(final_score)

    orchestrator_content = (
        f"[{s_color}]FINAL HOLISTIC SCORE:  {final_score} / 6[/{s_color}]\n\n"
        f"{score_bar(final_score, 6)}\n\n"
        f"[bold white]Orchestrator Feedback:[/bold white]\n\n"
        f"{results['final_feedback']}"
    )

    console.print(Panel(
        orchestrator_content,
        title="[bold green]🎯  Final Evaluation[/bold green]",
        border_style="green",
        padding=(1, 3),
        expand=True,
    ))

    console.print()
    console.rule("[dim]End of MAGIC Evaluation[/dim]")
    console.print()
