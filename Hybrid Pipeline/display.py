"""
display.py — Terminal display for Hybrid Pipeline results

Shows three stages sequentially:
  Stage 1 — SRCE evidence JSON (what every agent was anchored to)
  Stage 2 — 5 MAGIC agent scores + per-agent feedback
  Stage 3 — Orchestrator holistic score + pipeline audit line
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
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


def clean_feedback(text: str) -> str:
    """
    Strips raw JSON wrapper from orchestrator feedback.
    Handles multiple Ollama output patterns:
      1. Valid JSON:          {"score": 5, "examiner_comment": "..."}
      2. Unquoted value:      {"score": 5, "examiner_comment": The student...}
      3. Markdown header:     ### JSON Output:\n{\n  "score": 5, ...}
      4. Plain text:          returned as-is
    """
    import re, json
    stripped = text.strip()

    # Strip markdown headers like "### JSON Output:" that Ollama sometimes adds
    stripped = re.sub(r'^#+\s*JSON\s*Output\s*:\s*', '', stripped, flags=re.IGNORECASE).strip()

    # Attempt 1: valid JSON — cleanest path
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "examiner_comment" in obj:
            return str(obj["examiner_comment"]).strip()
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: targeted regex — extract everything after "examiner_comment":
    # handles both quoted and unquoted values, and trailing } or "}
    match = re.search(
        r'"examiner_comment"\s*:\s*"?(.*?)(?:"?\s*\}?\s*)$',
        stripped, re.DOTALL
    )
    if match:
        comment = match.group(1).strip().rstrip('"}').strip()
        if len(comment) > 20:   # sanity check — real comment is never <20 chars
            return comment

    # Attempt 3: if block starts with { drop all JSON-looking header lines
    if stripped.startswith("{"):
        lines = stripped.splitlines()
        clean_lines = []
        skip_pattern = re.compile(r'^\s*[\{\}]|^\s*"(score|examiner_comment)"\s*:|^\s*#+\s')
        past_header = False
        for line in lines:
            if not past_header and skip_pattern.match(line):
                continue
            past_header = True
            clean_lines.append(line)
        result = "\n".join(clean_lines).strip().rstrip('"}').strip()
        if result:
            return result

    # Fallback: return as-is
    return stripped


# ── Main display function ──────────────────────────────────────────────────────

def display_results(
    results: dict,
    backend: str = "gemini",
    model: str = "gemini-2.0-flash",
    example: str = "",
):
    """
    Renders the full Hybrid pipeline output to the terminal.

    Expected keys in results:
        - evidence_dict:      dict   (raw SRCE output)
        - evidence_json:      str    (pretty-printed JSON string)
        - agent_results:      list of AgentResult dicts (sorted by agent_index)
        - domain_scores:      list of 5 ints
        - orchestrator_score: int
        - final_feedback:     str
    """
    backend_color = "cyan" if backend == "ollama" else "blue"
    example_label = f" · [dim]essay: {example}[/dim]" if example else ""

    console.print()
    console.rule("[bold magenta] 🔬  HYBRID Pipeline — MAGIC + AutoSCORE  🔬 [/bold magenta]")
    console.print(
        f"  [{backend_color}]🔧 Backend:[/{backend_color}] [bold]{backend.upper()}[/bold]"
        f"   [{backend_color}]⚙  Model:[/{backend_color}] [bold]{model}[/bold]"
        + example_label
    )
    console.print(
        "  [dim]Architecture: SRCE Evidence Extraction → 5 Evidence-Anchored Agents → Orchestrator[/dim]"
    )
    console.print()

    # ── Stage 1: SRCE Evidence JSON ───────────────────────────────────────────
    console.rule("[bold cyan]◆  Stage 1 · SRCE Agent — Structured Evidence Record[/bold cyan]")
    console.print()
    console.print(
        "  [dim]Every agent below was anchored to this evidence record instead of re-reading the essay freely.[/dim]\n"
    )

    evidence_json = results.get("evidence_json", "{}")
    syntax = Syntax(evidence_json, "json", theme="monokai", line_numbers=False, word_wrap=True)
    console.print(Panel(
        syntax,
        title="[cyan]Evidence Record (SRCE Output)[/cyan]",
        border_style="cyan",
        padding=(1, 2),
        expand=True,
    ))
    console.print()

    # ── Stage 2: Agent Scores Summary Table ───────────────────────────────────
    console.rule("[bold blue]◆  Stage 2 · 5 MAGIC Agents (Evidence-Anchored, Parallel)[/bold blue]")
    console.print()
    console.print("[bold white]Agent Scores Summary[/bold white]\n")

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="dim blue",
        padding=(0, 1),
    )
    table.add_column("#",       style="dim",  width=4)
    table.add_column("Agent",   style="bold", width=14)
    table.add_column("Aspect",                width=52)
    table.add_column("Score",                 width=20)

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

    # ── Per-agent Feedback Panels ─────────────────────────────────────────────
    console.print("[bold white]Detailed Reasoning per Agent[/bold white]\n")

    for i, agent in enumerate(results["agent_results"]):
        color = ASPECT_COLORS[i]
        icon  = ASPECT_ICONS[i]
        title = (
            f"{icon} [bold {color}]Agent {i + 1}[/bold {color}]"
            f" · {agent['aspect_name']}"
            f"   {score_bar(agent['score'])}"
        )
        console.print(Panel(
            agent["examiner_comment"],
            title=title,
            border_style=color,
            padding=(1, 2),
            expand=True,
        ))
        console.print()

    # ── Score Summary Bar ─────────────────────────────────────────────────────
    console.rule("[bold white]Score Summary[/bold white]")
    console.print()

    summary = Table(
        box=box.SIMPLE,
        show_header=False,
        padding=(0, 2),
        border_style="dim",
    )
    summary.add_column(width=36)
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

    # ── Stage 3: Orchestrator Final Panel ─────────────────────────────────────
    console.rule("[bold green] 🎯  Stage 3 · Orchestrator — Final Holistic Score [/bold green]")
    console.print()

    final_score = results["orchestrator_score"]
    s_color     = score_color(final_score)

    orchestrator_content = (
        f"[{s_color}]FINAL HOLISTIC SCORE:  {final_score} / 6[/{s_color}]\n\n"
        f"{score_bar(final_score, 6)}\n\n"
        f"[bold white]Orchestrator Feedback:[/bold white]\n\n"
        f"{clean_feedback(results['final_feedback'])}"
    )

    console.print(Panel(
        orchestrator_content,
        title="[bold green]🎯  Hybrid Final Evaluation[/bold green]",
        border_style="green",
        padding=(1, 3),
        expand=True,
    ))

    # ── Pipeline Audit ────────────────────────────────────────────────────────
    console.print()
    console.rule("[dim]Pipeline Audit[/dim]")
    console.print(
        "[dim]SRCE extracted evidence → 5 agents scored anchored to that evidence → orchestrator synthesised[/dim]"
    )
    console.print(
        f"[dim]Architecture: MAGIC + AutoSCORE Hybrid  ·  Backend: {backend.upper()}  ·  Model: {model}[/dim]"
    )
    console.rule()
    console.print()
