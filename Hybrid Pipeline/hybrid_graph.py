"""
hybrid_graph.py — MAGIC + AutoSCORE Hybrid Pipeline (LangGraph)

Architecture:
    START
      └─► srce_node          (Agent 1: SRCE evidence extraction — sequential)
            └─► route_to_agents  (fan-out via Send())
                  └─► agent_node × 5   (parallel, each anchored to SRCE evidence JSON)
                        └─► orchestrator_node   (fan-in, synthesise holistic score)
                              └─► END

LLM calls per essay: 7  (1 SRCE + 5 agents + 1 orchestrator)
Context per agent:  ~1300 tokens  (+350 vs MAGIC for the evidence JSON)

Usage:
    # Gemini (default)
    python hybrid_graph.py

    # Gemini with specific model
    python hybrid_graph.py --backend gemini --model gemini-1.5-pro

    # Ollama with llama3.1
    python hybrid_graph.py --backend ollama --model llama3.1:8b

    # Choose a built-in essay example
    python hybrid_graph.py --example strong
    python hybrid_graph.py --example average
    python hybrid_graph.py --example weak

    # Combine flags
    python hybrid_graph.py --backend ollama --model llama3.1:8b --example weak
"""

import os
import json
import re
import operator
import argparse
from typing import Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict

import llm_client
from prompts.agents import GREAgentPrompts
from prompts.orchestrator import GREOrchestratorPrompts
from prompts.srce_prompt import format_srce_prompt

# ── Load environment ───────────────────────────────────────────────────────────
load_dotenv()


# ── Rubric (combined — passed to SRCE agent so it extracts across all 5 traits)
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


# ── JSON parsing helper (same as AutoSCORE) ────────────────────────────────────

def parse_evidence_json(raw: str, label: str) -> dict:
    """
    Parses JSON from SRCE agent raw response.
    Handles markdown fences, leading text, truncation, and common LLM JSON errors.
    """
    # Strip markdown fences
    cleaned = llm_client.strip_markdown_fences(raw) if hasattr(llm_client, "strip_markdown_fences") \
              else re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()

    # Attempt 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract outermost { ... }
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Attempt 3: JSON repair — fix common LLM output issues
    repaired = cleaned
    json_start = repaired.find('{')
    if json_start >= 0:
        repaired = repaired[json_start:]
    # Fix trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    # If JSON was truncated (unclosed braces), close them
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        # Close any open string, then close braces
        if repaired.rstrip()[-1] not in ('}', ']', '"', 'e', 'l'):
            repaired += '"'
        repaired += '}' * open_braces
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Attempt 4: tolerant repair via json-repair.
    # Qwen2.5 frequently emits invalid JSON the regex repairs above can't fix:
    #   • single-quoted string values  ('...')
    #   • unescaped double-quotes inside a string value
    #     (e.g. "position_quote": "...for "Driverless Cars" i am against...")
    # json-repair recovers both while preserving the nested trait structure.
    try:
        import json_repair
        result = json_repair.loads(cleaned)
        if isinstance(result, dict) and result:
            return result
    except Exception:
        pass

    print(f"  [!] JSON parse failed for {label}. Raw length: {len(raw)} chars")
    print(f"      Snippet: {raw[:300]}")
    return {}


# ── State Definitions ──────────────────────────────────────────────────────────

class AgentResult(TypedDict):
    agent_index:      int
    aspect_name:      str
    agent_type:       str
    score:            int
    examiner_comment: str


class HybridState(TypedDict):
    essay_text:         str
    essay_prompt:       str
    rubric:             str
    evidence_dict:      dict   # populated by srce_node
    evidence_json:      str    # passed to all 5 agent_nodes
    # operator.add safely merges results from all 5 parallel agents
    agent_results:      Annotated[list[AgentResult], operator.add]
    orchestrator_score: int
    final_feedback:     str


class AgentSubState(TypedDict):
    """Payload sent to each individual agent node via Send()"""
    essay_text:    str
    essay_prompt:  str
    agent_index:   int
    agent_type:    str
    rubric_text:   str
    aspect_name:   str
    evidence_json: str   # ← NEW vs. MAGIC: evidence passed to every agent


# ── LangGraph Nodes ────────────────────────────────────────────────────────────

def srce_node(state: HybridState) -> dict:
    """
    Stage 1 — SRCE evidence extraction (AutoSCORE Agent 1).
    Runs sequentially BEFORE the 5 MAGIC agents.
    Extracts structured JSON evidence from the essay using the full combined rubric.
    """
    print("\n  [SRCE] Extracting structured evidence from essay...")

    system_prompt, user_prompt = format_srce_prompt(
        essay_text=state["essay_text"],
        essay_prompt=state["essay_prompt"],
        rubric=state["rubric"],
    )

    raw           = llm_client.call_llm_raw(system_prompt, user_prompt, "SRCE Agent")
    evidence_dict = parse_evidence_json(raw, "SRCE Agent")
    evidence_json = json.dumps(evidence_dict, indent=2)

    trait_count = len([k for k in evidence_dict if k != "essay_metadata"])
    print(f"  [SRCE] ✓ Evidence extracted — {trait_count}/5 trait fields populated")

    return {
        "evidence_dict": evidence_dict,
        "evidence_json": evidence_json,
    }


def agent_node(state: AgentSubState) -> dict:
    """
    Stage 2 — Single specialist MAGIC agent (runs in parallel with other 4).
    Receives essay + SRCE evidence JSON → scores its assigned trait.
    THE KEY HYBRID CHANGE: evidence_json is passed to format_prompt_inference().
    """
    grading_instruction = {
        "essay_text": state["essay_text"],
        "prompt":     state["essay_prompt"],
    }

    formatted_prompt = GREAgentPrompts.format_prompt_inference(
        grading_instruction=grading_instruction,
        agent_rubric_type=state["agent_type"],
        current_aspect_rubric=state["rubric_text"],
        evidence_json=state["evidence_json"],   # ← THE HYBRID CHANGE
    )

    result = llm_client.call_llm(formatted_prompt, state["aspect_name"])

    agent_result: AgentResult = {
        "agent_index":      state["agent_index"],
        "aspect_name":      state["aspect_name"],
        "agent_type":       state["agent_type"],
        "score":            result["score"],
        "examiner_comment": result["examiner_comment"],
    }

    return {"agent_results": [agent_result]}


def orchestrator_node(state: HybridState) -> dict:
    """
    Stage 3 — MAGIC Orchestrator.
    Runs after all 5 agents complete (fan-in via operator.add).
    Sorts by agent_index (parallel nodes arrive in non-deterministic order),
    then synthesises a holistic score and feedback.
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

    result = llm_client.call_llm_orchestrator(orchestrator_prompt, "Orchestrator")

    return {
        "orchestrator_score": result["score"],
        "final_feedback":     result["examiner_comment"],
    }


# ── Router: Fan-out to 5 Parallel Agents (after SRCE completes) ───────────────

def route_to_agents(state: HybridState) -> list[Send]:
    """
    Conditional edge from srce_node → 5 × agent_node (parallel).
    Passes evidence_json from SRCE into every AgentSubState.
    """
    return [
        Send("agent_node", {
            "essay_text":    state["essay_text"],
            "essay_prompt":  state["essay_prompt"],
            "agent_index":   i,
            "agent_type":    rubric_type,
            "rubric_text":   rubric_text,
            "aspect_name":   aspect_name,
            "evidence_json": state["evidence_json"],   # ← injected into every agent
        })
        for i, (rubric_type, rubric_text, aspect_name)
        in enumerate(GREAgentPrompts.aspect_rubrics)
    ]


# ── Build the LangGraph ────────────────────────────────────────────────────────

def build_graph():
    """
    3-stage graph:
        START → srce_node → [route_to_agents] → agent_node × 5 → orchestrator_node → END
    """
    graph = StateGraph(HybridState)

    graph.add_node("srce_node",         srce_node)
    graph.add_node("agent_node",        agent_node)
    graph.add_node("orchestrator_node", orchestrator_node)

    # START → SRCE (sequential — agents need its output)
    graph.add_edge(START, "srce_node")

    # SRCE → fan-out → 5 × agent_node (parallel)
    graph.add_conditional_edges("srce_node", route_to_agents, ["agent_node"])

    # All 5 agent_nodes fan-in → orchestrator (operator.add merges agent_results)
    graph.add_edge("agent_node", "orchestrator_node")

    graph.add_edge("orchestrator_node", END)

    return graph.compile()


# ── CLI Argument Parser ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Essay Evaluation Pipeline (MAGIC + AutoSCORE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hybrid_graph.py
  python hybrid_graph.py --backend gemini --model gemini-1.5-pro
  python hybrid_graph.py --backend ollama --model llama3.1:8b
  python hybrid_graph.py --backend ollama --model gemma3:27b --example weak
  python hybrid_graph.py --example average
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
    parser.add_argument(
        "--essay-file",
        type=str,
        help="Path to a text file containing the student essay to evaluate",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Path to a text file containing the essay prompt/question (optional)",
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

    print(f"\n  [Hybrid] Backend:  {args.backend.upper()}  |  Model: {model}")

    if args.essay_file:
        with open(args.essay_file, "r", encoding="utf-8") as f:
            essay_text = f.read().strip()
        if args.prompt_file:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                essay_prompt = f.read().strip()
        else:
            essay_prompt = ESSAY_PROMPT.strip()
            print("  [Hybrid] Warning: No --prompt-file provided, using default GRE prompt.")
        print(f"  [Hybrid] Essay:    Loaded from '{args.essay_file}'")
    else:
        essay_text = ESSAYS[args.example].strip()
        essay_prompt = ESSAY_PROMPT.strip()
        print(f"  [Hybrid] Essay:    '{args.example}' example")

    print(f"\n🚀 Starting Hybrid pipeline — SRCE → 5 agents (parallel) → orchestrator...\n")

    app         = build_graph()
    final_state = app.invoke({
        "essay_text":         essay_text,
        "essay_prompt":       essay_prompt,
        "rubric":             RUBRIC.strip(),
        "evidence_dict":      {},
        "evidence_json":      "",
        "agent_results":      [],
        "orchestrator_score": 0,
        "final_feedback":     "",
    })

    # Sort results by agent_index for deterministic display order
    sorted_results = sorted(final_state["agent_results"], key=lambda r: r["agent_index"])

    display_results(
        results={
            "evidence_dict":      final_state["evidence_dict"],
            "evidence_json":      final_state["evidence_json"],
            "agent_results":      sorted_results,
            "domain_scores":      [r["score"] for r in sorted_results],
            "orchestrator_score": final_state["orchestrator_score"],
            "final_feedback":     final_state["final_feedback"],
        },
        backend=args.backend,
        model=model,
        example=args.example,
    )
