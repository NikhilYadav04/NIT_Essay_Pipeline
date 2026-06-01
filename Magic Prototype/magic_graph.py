"""
magic_graph.py — MAGIC Pipeline (LangGraph + Dynamic Backend)

Runs 5 specialist agents in PARALLEL via LangGraph Send(),
fans-in to the orchestrator, then renders a Rich terminal UI.

Supports Gemini (cloud) and Ollama (local 8b–27b models).

Usage:
    # Gemini (default)
    python magic_graph.py

    # Gemini with specific model
    python magic_graph.py --backend gemini --model gemini-1.5-pro

    # Ollama with llama3.1
    python magic_graph.py --backend ollama --model llama3.1:8b

    # Ollama with gemma3
    python magic_graph.py --backend ollama --model gemma3:27b

    # Choose a built-in essay example
    python magic_graph.py --example strong
    python magic_graph.py --example average
    python magic_graph.py --example weak

    # Combine flags
    python magic_graph.py --backend ollama --model llama3.1:8b --example weak
"""

import os
import operator
import argparse
from typing import Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict

import llm_client
from prompts.agents import GREAgentPrompts
from prompts.agents_asap import ASAPAgentPrompts
from prompts.orchestrator import GREOrchestratorPrompts

# ── Load environment ───────────────────────────────────────────────────────────
load_dotenv()

# Active prompts class — swapped by build_graph(rubric=...)
_AGENT_PROMPTS = GREAgentPrompts


# ── Essay Prompt (shared for all examples) ─────────────────────────────────────
ESSAY_PROMPT = """
The following appeared in an article about urban planning:
"Cities that have implemented bike-sharing programs have seen a significant reduction
in traffic congestion. Therefore, our city should implement a bike-sharing program
to reduce traffic congestion."
Write a response in which you examine the stated and/or unstated assumptions of the
argument. Be sure to explain how the argument depends on these assumptions and what
the implications are for the argument if the assumptions prove unwarranted.
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
cycling infrastructure quality, trip distances, and cultural attitudes—none of which the article
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


# ── State Definitions ──────────────────────────────────────────────────────────

class AgentResult(TypedDict):
    agent_index:      int
    aspect_name:      str
    agent_type:       str
    score:            int
    examiner_comment: str


class MAGICState(TypedDict):
    essay_text:        str
    essay_prompt:      str
    # operator.add safely merges results from all 5 parallel agents
    agent_results:     Annotated[list[AgentResult], operator.add]
    orchestrator_score: int
    final_feedback:    str


class AgentSubState(TypedDict):
    """Payload sent to each individual agent node via Send()"""
    essay_text:   str
    essay_prompt: str
    agent_index:  int
    agent_type:   str
    rubric_text:  str
    aspect_name:  str


# ── LangGraph Nodes ────────────────────────────────────────────────────────────

def agent_node(state: AgentSubState) -> dict:
    """
    Single specialist agent node — runs in parallel with the other 4.
    Formats its own rubric prompt, calls LLM, returns one AgentResult.
    """
    grading_instruction = {
        "essay_text": state["essay_text"],
        "prompt":     state["essay_prompt"],
    }

    formatted_prompt = _AGENT_PROMPTS.format_prompt_inference(
        grading_instruction=grading_instruction,
        agent_rubric_type=state["agent_type"],
        current_aspect_rubric=state["rubric_text"],
    )

    result = llm_client.call_llm(formatted_prompt, state["aspect_name"])

    agent_result: AgentResult = {
        "agent_index":      state["agent_index"],
        "aspect_name":      state["aspect_name"],
        "agent_type":       state["agent_type"],
        "score":            result["score"],
        "examiner_comment": result["examiner_comment"],
    }

    # Return as list — operator.add appends this to the shared accumulator
    return {"agent_results": [agent_result]}


def orchestrator_node(state: MAGICState) -> dict:
    """
    Runs after all 5 agents complete.
    Sorts by agent_index (parallel nodes arrive in non-deterministic order),
    then synthesises a final holistic score.
    """
    sorted_results = sorted(state["agent_results"], key=lambda r: r["agent_index"])

    domain_scores    = [r["score"]            for r in sorted_results]
    domain_feedbacks = [r["examiner_comment"] for r in sorted_results]

    grading_instruction = {
        "essay_text": state["essay_text"],
        "prompt":     state["essay_prompt"],
    }

    orchestrator_prompt = GREOrchestratorPrompts.format_prompt_inference(
        grading_instruction=grading_instruction,
        domain_scores=domain_scores,
        domain_feedbacks=domain_feedbacks,
    )

    result = llm_client.call_llm(orchestrator_prompt, "Orchestrator")

    return {
        "orchestrator_score": result["score"],
        "final_feedback":     result["examiner_comment"],
    }


# ── Router: Fan-out to 5 Parallel Agents ──────────────────────────────────────

def route_to_agents(state: MAGICState) -> list[Send]:
    """
    Conditional edge from START.
    Returns 5 Send() objects — LangGraph executes all 5 agent_node calls in parallel.
    """
    return [
        Send("agent_node", {
            "essay_text":   state["essay_text"],
            "essay_prompt": state["essay_prompt"],
            "agent_index":  i,
            "agent_type":   rubric_type,
            "rubric_text":  rubric_text,
            "aspect_name":  aspect_name,
        })
        for i, (rubric_type, rubric_text, aspect_name)
        in enumerate(_AGENT_PROMPTS.aspect_rubrics)
    ]


# ── Build the LangGraph ────────────────────────────────────────────────────────

def build_graph(rubric: str = "gre"):
    global _AGENT_PROMPTS
    _AGENT_PROMPTS = ASAPAgentPrompts if rubric == "asap" else GREAgentPrompts

    graph = StateGraph(MAGICState)

    graph.add_node("agent_node",        agent_node)
    graph.add_node("orchestrator_node", orchestrator_node)

    # START → fan-out → 5 × agent_node (parallel)
    graph.add_conditional_edges(START, route_to_agents, ["agent_node"])

    # All 5 agent_nodes fan-in here (operator.add merges agent_results)
    graph.add_edge("agent_node", "orchestrator_node")

    graph.add_edge("orchestrator_node", END)

    return graph.compile()


# ── CLI Argument Parser ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="MAGIC Essay Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python magic_graph.py
  python magic_graph.py --backend gemini --model gemini-1.5-pro
  python magic_graph.py --backend ollama --model llama3.1:8b
  python magic_graph.py --backend ollama --model gemma3:27b --example weak
  python magic_graph.py --example strong
        """,
    )

    parser.add_argument(
        "--backend",
        choices=["gemini", "ollama"],
        default="gemini",
        help="LLM backend to use (default: gemini)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name. Defaults: gemini→'gemini-2.0-flash', ollama→'llama3.1:8b'\n"
            "Gemini options: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro\n"
            "Ollama options: llama3.1:8b, gemma3:12b, gemma3:27b, mistral:7b, phi4:14b"
        ),
    )
    parser.add_argument(
        "--example",
        choices=["strong", "average", "weak"],
        default="strong",
        help="Which built-in essay example to evaluate (default: strong)",
    )

    return parser.parse_args()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from display import display_results

    args = parse_args()

    # Resolve default model per backend
    default_models = {"gemini": "gemini-2.0-flash", "ollama": "llama3.1:8b"}
    model = args.model or default_models[args.backend]

    # Configure the LLM client
    api_key = os.environ.get("GOOGLE_API_KEY") if args.backend == "gemini" else None
    llm_client.configure(backend=args.backend, model=model, api_key=api_key)

    essay_text = ESSAYS[args.example].strip()

    print(f"\n  [MAGIC] Backend:  {args.backend.upper()}  |  Model: {model}")
    print(f"  [MAGIC] Essay:    '{args.example}' example")
    print(f"\n🚀 Starting MAGIC pipeline — 5 agents running in parallel...\n")

    # Build and run the graph
    app         = build_graph()
    final_state = app.invoke({
        "essay_text":    essay_text,
        "essay_prompt":  ESSAY_PROMPT.strip(),
        "agent_results": [],
    })

    # Sort results by agent_index for deterministic display order
    sorted_results = sorted(final_state["agent_results"], key=lambda r: r["agent_index"])

    display_results(
        results={
            "agent_results":      sorted_results,
            "domain_scores":      [r["score"] for r in sorted_results],
            "orchestrator_score": final_state["orchestrator_score"],
            "final_feedback":     final_state["final_feedback"],
        },
        backend=args.backend,
        model=model,
        example=args.example,
    )
