"""
autoscore_graph.py — AutoSCORE Two-Agent Sequential Pipeline (LangGraph)

Architecture:
    START → srce_node (Agent 1: evidence extraction)
          → scoring_node (Agent 2: rubric judgment from evidence)
          → END → display

Agent 1 reads the essay and outputs structured JSON evidence.
Agent 2 reads the JSON evidence (not the essay freely) and assigns scores.
The JSON intermediate is the auditability guarantee.

Usage:
    python autoscore_graph.py
    python autoscore_graph.py --backend gemini --model gemini-1.5-pro
    python autoscore_graph.py --backend ollama --model llama3.1:8b
    python autoscore_graph.py --example weak
    python autoscore_graph.py --backend ollama --model gemma3:27b --example average
"""

import os
import sys
import json
import argparse

# Add shared/ to path so we can import llm_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "shared"))

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

import llm_client
from prompts.srce_prompt import format_srce_prompt
from prompts.scoring_prompt import format_scoring_prompt

# ── Load environment ───────────────────────────────────────────────────────────
load_dotenv()


# ── Rubric ─────────────────────────────────────────────────────────────────────
RUBRIC = """
Trait 1: Task Response
  Score 6: Clear, insightful position directly addressing the prompt
  Score 5: Clear, well-considered position addressing the prompt
  Score 4: Clear position addressing the prompt
  Score 3: Vague or limited in addressing the prompt
  Score 2: Unclear or seriously limited response to the prompt
  Score 1: Little or no understanding of how to respond
  Score 0: Off-topic, foreign language, copied topic, or illegible

Trait 2: Argument Quality
  Score 6: Fully developed with compelling reasons and persuasive examples
  Score 5: Developed with logically sound reasons and well-chosen examples
  Score 4: Developed with relevant reasons and/or examples
  Score 3: Weak use of reasons/examples, relies on unsupported claims
  Score 2: Few or no relevant reasons or examples
  Score 1: Little or no evidence of understanding the issue
  Score 0: Off-topic, foreign language, copied topic, or illegible

Trait 3: Organisation
  Score 6: Well-focused, well-organised, connecting ideas logically
  Score 5: Focused and generally well-organised
  Score 4: Adequately focused and organised
  Score 3: Limited in focus and/or organisation
  Score 2: Poorly focused and/or poorly organised
  Score 1: Little or no organised response
  Score 0: Off-topic, foreign language, copied topic, or illegible

Trait 4: Language & Style
  Score 6: Fluent and precise, effective vocabulary and sentence variety
  Score 5: Clear and well-expressed, appropriate vocabulary and variety
  Score 4: Acceptable clarity, sufficient language control
  Score 3: Language problems causing lack of clarity
  Score 2: Serious language problems frequently interfering with meaning
  Score 1: Severe language problems persistently interfering with meaning
  Score 0: Off-topic, foreign language, copied topic, or illegible

Trait 5: Grammar & Mechanics
  Score 6: Superior facility with grammar — minor errors only
  Score 5: Good facility with conventions — minor errors only
  Score 4: General control of conventions — some errors
  Score 3: Occasional major or frequent minor errors interfering with meaning
  Score 2: Serious errors frequently obscuring meaning
  Score 1: Pervasive errors resulting in incoherence
  Score 0: Off-topic, foreign language, copied topic, or illegible
"""

# ── Essay Prompt ───────────────────────────────────────────────────────────────
ESSAY_PROMPT = """
The following appeared in an article about urban planning:
'Cities that have implemented bike-sharing programs have seen a significant reduction
in traffic congestion. Therefore, our city should implement a bike-sharing program
to reduce traffic congestion.'
Write a response examining the stated and/or unstated assumptions of the argument.
"""

# ── Built-in Example Essays ────────────────────────────────────────────────────
ESSAYS = {
    "strong": """
The argument presented in the article rests on a series of unverified assumptions that significantly
weaken its conclusion. While the proposal to implement a bike-sharing program to reduce traffic
congestion may ultimately be sound, the reasoning offered fails to substantiate this claim rigorously.

The most critical assumption is that of causation. The article asserts that cities with bike-sharing
programs have experienced reduced congestion, but this correlation does not establish that the program
caused the reduction. Numerous confounding variables—improved public transit, remote work adoption,
urban densification policies, or economic downturns reducing car usage—could equally explain the
observed decline. Without a controlled study isolating the effect of the bike-sharing program, the
causal link remains speculative.

A second problematic assumption is that of transferability. The argument implicitly presumes that
what succeeded in other cities will succeed in ours. However, cities differ vastly in topography,
climate, population density, commuter distances, and cycling culture. Programs that thrived in
Amsterdam or Bogotá may be irrelevant to a sprawling, car-centric city with inadequate cycling
infrastructure or extreme weather conditions. The argument would be significantly strengthened by
demonstrating that our city shares the key characteristics of the cited success cases.

Furthermore, the argument assumes sufficient resident adoption. A bike-sharing program can only
reduce congestion if enough residents switch from cars to bikes. This depends on perceived safety,
cycling infrastructure quality, trip distances, and cultural attitudes — none of which the article
addresses. Without evidence of likely adoption rates, the assumed outcome remains unsubstantiated.

Finally, the argument ignores opportunity costs and potential unintended consequences. Funds allocated
to a bike-sharing program could alternatively improve public transit or road infrastructure. Moreover,
adding cyclists to roads without dedicated lanes may increase accidents or paradoxically worsen
traffic flow. A responsible argument would weigh these trade-offs explicitly.

In sum, the argument's conclusion may be achievable, but only after addressing these foundational
assumptions with city-specific data, causal evidence, and comparative analysis.
""",

    "average": """
The argument says that because other cities reduced traffic with bike-sharing programs, our city
should do the same. This makes some sense but there are a few problems with the reasoning.

First, we don't know if the bike-sharing program actually caused the traffic reduction. Maybe
traffic went down for other reasons, like more people working from home. The article doesn't
prove that the bikes were responsible.

Second, our city might be different from the cities mentioned. If our city is very spread out
or doesn't have good cycling paths, people might not use the bikes even if we buy them. The
argument doesn't consider this.

Third, the argument assumes people will actually want to use the bikes. Some people prefer cars
because it's faster or more comfortable. Without knowing if residents would actually switch to
bikes, we can't be sure the program would work.

Overall the argument has some logic to it but relies on too many assumptions. It would be more
convincing if it included data specific to our city and explained why the comparison to other
cities is valid.
""",

    "weak": """
I think the city should get bikes because bikes are good. Other cities did it and traffic went
down so we should do it too. It makes sense because if people ride bikes they won't drive cars.

There might be some problems but overall it is a good idea. The city should just try it and
see what happens. Bikes are also good for the environment and health so there are many reasons
to do it.
""",
}


# ── LangGraph State ────────────────────────────────────────────────────────────

class AutoSCOREState(TypedDict):
    essay_text:    str
    essay_prompt:  str
    rubric:        str
    evidence_dict: dict   # populated by srce_node
    evidence_json: str    # stringified JSON, passed to scoring_node
    scoring_dict:  dict   # populated by scoring_node


# ── JSON parsing helper ────────────────────────────────────────────────────────

def parse_json(raw: str, label: str) -> dict:
    """
    Parses JSON from LLM raw response.
    Handles markdown fences, leading text, and partial wrapping.
    """
    # Strip markdown fences
    cleaned = llm_client.strip_markdown_fences(raw)

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find outermost JSON object
    import re
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print(f"  [!] Could not parse JSON for {label}. Raw snippet:")
    print(f"      {raw[:200]}...")
    return {}


# ── LangGraph Nodes ────────────────────────────────────────────────────────────

def srce_node(state: AutoSCOREState) -> dict:
    """
    Agent 1 — SRCE (Scoring Rubric Component Extraction).
    Reads essay + rubric → outputs structured evidence JSON.
    Does NOT assign scores.
    """
    system_prompt, user_prompt = format_srce_prompt(
        essay_text=state["essay_text"],
        essay_prompt=state["essay_prompt"],
        rubric=state["rubric"],
    )

    raw           = llm_client.call_llm_raw(system_prompt, user_prompt, "SRCE Agent")
    evidence_dict = parse_json(raw, "SRCE Agent")
    evidence_json = json.dumps(evidence_dict, indent=2)

    return {
        "evidence_dict": evidence_dict,
        "evidence_json": evidence_json,
    }


def scoring_node(state: AutoSCOREState) -> dict:
    """
    Agent 2 — Scoring Agent (SA).
    Reads evidence JSON (primary) + rubric → assigns per-trait scores + holistic.
    Essay provided as reference only — Agent 2 is anchored to the JSON.
    """
    system_prompt, user_prompt = format_scoring_prompt(
        essay_text=state["essay_text"],
        essay_prompt=state["essay_prompt"],
        rubric=state["rubric"],
        evidence_json=state["evidence_json"],
    )

    raw          = llm_client.call_llm_raw(system_prompt, user_prompt, "Scoring Agent")
    scoring_dict = parse_json(raw, "Scoring Agent")

    return {"scoring_dict": scoring_dict}


# ── Build the Graph ────────────────────────────────────────────────────────────

def build_graph():
    """
    Linear sequential graph:
        START → srce_node → scoring_node → END
    """
    graph = StateGraph(AutoSCOREState)

    graph.add_node("srce_node",    srce_node)
    graph.add_node("scoring_node", scoring_node)

    graph.add_edge(START,          "srce_node")
    graph.add_edge("srce_node",    "scoring_node")
    graph.add_edge("scoring_node", END)

    return graph.compile()


# ── CLI Argument Parser ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoSCORE — Two-Agent Essay Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autoscore_graph.py
  python autoscore_graph.py --backend gemini --model gemini-1.5-pro
  python autoscore_graph.py --backend ollama --model llama3.1:8b
  python autoscore_graph.py --backend ollama --model gemma3:27b --example weak
  python autoscore_graph.py --example average
        """,
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "ollama"],
        default="gemini",
        help="LLM backend (default: gemini)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name. Defaults: gemini→'gemini-2.0-flash', ollama→'llama3.1:8b'\n"
            "Gemini: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro\n"
            "Ollama: llama3.1:8b, gemma3:12b, gemma3:27b, mistral:7b, phi4:14b"
        ),
    )
    parser.add_argument(
        "--example",
        choices=["strong", "average", "weak"],
        default="strong",
        help="Which built-in essay to evaluate (default: strong)",
    )
    return parser.parse_args()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from display import display_results

    args = parse_args()

    # Resolve default model per backend
    default_models = {"gemini": "gemini-2.0-flash", "ollama": "llama3.1:8b"}
    model   = args.model or default_models[args.backend]
    api_key = os.environ.get("GOOGLE_API_KEY") if args.backend == "gemini" else None

    # Configure LLM backend
    llm_client.configure(backend=args.backend, model=model, api_key=api_key)

    essay_text = ESSAYS[args.example].strip()

    print(f"\n  [AutoSCORE] Backend: {args.backend.upper()}  |  Model: {model}")
    print(f"  [AutoSCORE] Essay:   '{args.example}' example")
    print("\n🚀 Starting AutoSCORE pipeline...")
    print("   Step 1 → SRCE Agent extracting evidence...")

    # Build and run
    app         = build_graph()
    final_state = app.invoke({
        "essay_text":    essay_text,
        "essay_prompt":  ESSAY_PROMPT.strip(),
        "rubric":        RUBRIC.strip(),
        "evidence_dict": {},
        "evidence_json": "",
        "scoring_dict":  {},
    })

    print("   Step 2 → Scoring Agent applying rubric judgment...")
    print("   Done.\n")

    display_results(
        results={
            "evidence_dict": final_state["evidence_dict"],
            "evidence_json": final_state["evidence_json"],
            "scoring_dict":  final_state["scoring_dict"],
        },
        backend=args.backend,
        model=model,
        example=args.example,
    )
