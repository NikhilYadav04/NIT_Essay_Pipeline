# MAGIC Essay Evaluation — LangGraph Implementation Doc

> **Multi-Agent Grading with Iterative Critique (MAGIC)**  
> Prototype using Google Gemini + LangGraph for parallel multi-agent essay scoring.

---

## 1. Overview

MAGIC is a multi-agent pipeline that evaluates student essays across 5 specialised dimensions, then synthesises those scores into a final holistic grade using an orchestrator agent.

**Input:** Essay text (string) + Essay prompt (string)  
**Output:** 5 agent scores + reasoning + 1 final orchestrator score + combined feedback

---

## 2. Architecture Diagram

```
                    ┌─────────────────────┐
                    │      START NODE      │
                    │  Load essay + prompt │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   route_to_agents   │
                    │  Send() × 5 agents  │  ← LangGraph fan-out
                    └──────────┬──────────┘
                               │
          ┌──────────┬─────────┼─────────┬──────────┐
          │          │         │         │          │
    ┌─────▼───┐ ┌────▼────┐ ┌─▼──────┐ ┌▼───────┐ ┌▼───────┐
    │ Agent 1 │ │ Agent 2 │ │Agent 3 │ │Agent 4 │ │Agent 5 │
    │  Arg.   │ │  Arg.   │ │  Arg.  │ │ Vocab. │ │Grammar │
    └─────┬───┘ └────┬────┘ └─┬──────┘ └┬───────┘ └┬───────┘
          │          │         │         │          │
          └──────────┴─────────┼─────────┴──────────┘
                               │  fan-in (all 5 done)
                    ┌──────────▼──────────┐
                    │    ORCHESTRATOR      │
                    │  Synthesise 5 scores │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │      DISPLAY         │
                    │  Rich terminal UI    │
                    └─────────────────────┘
```

---

## 3. File Structure

```
magic_prototype/
├── docs/
│   └── implementation.md       ← this file
├── prompts/
│   ├── __init__.py             ← empty, makes it a package
│   ├── base.py                 ← shared prompts (Alpaca wrapper, output_format, rubric)
│   ├── agents.py               ← 5 agent rubrics + 3 system prompts + format_prompt_inference()
│   └── orchestrator.py         ← orchestrator system prompt + format_prompt_inference()
├── magic_graph.py              ← LangGraph pipeline (main entry point)
├── display.py                  ← Rich terminal UI formatter
└── .env                        ← GOOGLE_API_KEY=...
```

---

## 4. The Prompt System (prompts/)

### 4.1 `base.py` — `BasePrompts`

| Attribute | Purpose |
|---|---|
| `input_prompt` | Wraps essay in `<student_essay>` tags |
| `output_format` | Instructs model to reply with `{"score": X, "examiner_comment": "..."}` |
| `alpaca_prompt` | Alpaca instruction wrapper: `### Instruction / ### Input / ### Response` |
| `rubric` | Full holistic GRE rubric (Score 0–6 descriptors) |

### 4.2 `agents.py` — `GREAgentPrompts(BasePrompts)`

| Attribute | Purpose |
|---|---|
| `aspect_1_rubric` through `aspect_5_rubric` | Per-aspect scoring rubric strings |
| `aspect_rubrics` | List of 5 tuples: `(agent_type, rubric_text, aspect_name)` |
| `argumentative_system_prompt` | System prompt for agents 1, 2, 3 |
| `vocabulary_system_prompt` | System prompt for agent 4 |
| `grammar_system_prompt` | System prompt for agent 5 |
| `format_prompt_inference()` | Builds complete Alpaca-formatted prompt for a given agent |

**`format_prompt_inference(grading_instruction, agent_rubric_type, current_aspect_rubric)`**
- Selects correct system prompt based on `agent_rubric_type` (`"argumentative"`, `"vocabulary"`, `"grammar"`)
- Injects rubric, essay prompt, and output format into system prompt
- Wraps essay text in `input_prompt`
- Returns final Alpaca-wrapped string ready to send to LLM

### 4.3 `orchestrator.py` — `GREOrchestratorPrompts(BasePrompts)`

**`format_prompt_inference(grading_instruction, domain_scores, domain_feedbacks)`**
- Takes 5 scores + 5 feedback strings
- Formats them as `<expert_grader_judgement>` XML blocks
- Wraps in orchestrator system prompt + Alpaca format
- Returns final prompt string ready to send to LLM

---

## 5. The 5 Agents — Isolation Principle

Each agent sees **only its own rubric**. They are completely independent.

| Agent | Type | Aspect Scored |
|---|---|---|
| Agent 1 | `argumentative` | Aspect 1: Quality of response to the prompt |
| Agent 2 | `argumentative` | Aspect 2: Considering complexities of the issue |
| Agent 3 | `argumentative` | Aspect 3: Organising, developing, expressing ideas |
| Agent 4 | `vocabulary` | Aspect 4: Vocabulary and sentence variety |
| Agent 5 | `grammar` | Aspect 5: Grammar and mechanics |

> **Critical:** Agents never see each other's scores or reasoning. Only the orchestrator has full visibility.

---

## 6. LangGraph State Schema

```python
from typing import TypedDict, Annotated
import operator

class AgentResult(TypedDict):
    agent_index: int
    aspect_name: str
    agent_type: str
    score: int
    examiner_comment: str

class MAGICState(TypedDict):
    essay_text: str
    essay_prompt: str
    # Annotated with operator.add so parallel agents append safely
    agent_results: Annotated[list[AgentResult], operator.add]
    orchestrator_score: int
    final_feedback: str
```

---

## 7. LangGraph Node Definitions

### 7.1 `route_to_agents` (conditional edge / router)

```python
def route_to_agents(state: MAGICState) -> list[Send]:
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
        in enumerate(GREAgentPrompts.aspect_rubrics)
    ]
```

This sends **5 parallel `Send()` messages** — LangGraph runs them concurrently.

### 7.2 `agent_node`

```python
def agent_node(state: AgentSubState) -> dict:
    # 1. Build prompt via GREAgentPrompts.format_prompt_inference()
    # 2. Call Gemini API
    # 3. Parse JSON response
    # 4. Return {"agent_results": [AgentResult(...)]}
```

Returns a **list with one item** — LangGraph's `operator.add` reducer accumulates all 5.

### 7.3 `orchestrator_node`

```python
def orchestrator_node(state: MAGICState) -> dict:
    # 1. Sort agent_results by agent_index (ensure consistent order)
    # 2. Build prompt via GREOrchestratorPrompts.format_prompt_inference()
    # 3. Call Gemini API
    # 4. Parse JSON response
    # 5. Return {"orchestrator_score": int, "final_feedback": str}
```

### 7.4 Graph Assembly

```python
graph = StateGraph(MAGICState)
graph.add_node("agent_node", agent_node)
graph.add_node("orchestrator_node", orchestrator_node)

graph.add_conditional_edges(START, route_to_agents, ["agent_node"])
graph.add_edge("agent_node", "orchestrator_node")
graph.add_edge("orchestrator_node", END)

app = graph.compile()
```

---

## 8. LLM Call Helper — `call_llm()`

```python
def call_llm(prompt: str, label: str) -> dict:
    """
    Sends Alpaca-formatted prompt to Gemini.
    Returns {"score": int, "examiner_comment": str}
    Handles JSON extraction with regex fallback.
    """
```

- Model: `gemini-2.0-flash` (free tier, fast)
- JSON extracted via `re.search(r'\{.*?\}', raw, re.DOTALL)`
- Fallback: manual regex for score if JSON parse fails
- Error state: returns `{"score": 0, "examiner_comment": "Error: ..."}` — never crashes

---

## 9. Import Fixes Required

The original prompt files use relative imports that break when run as a standalone package.

| File | Original | Fixed |
|---|---|---|
| `agents.py` line 1 | `from .base import BasePrompts` | `from prompts.base import BasePrompts` |
| `orchestrator.py` line 1 | `from .base import BasePrompts` | `from prompts.base import BasePrompts` |
| `orchestrator.py` line 2 | `from .agents import GREAgentPrompts` | `from prompts.agents import GREAgentPrompts` |

---

## 10. Display Layer — `display.py`

Uses the `rich` library for a premium terminal UI:

| Section | What it shows |
|---|---|
| Agent score table | 5 agents × (aspect name, visual score bar) |
| Per-agent panels | Full `examiner_comment` in coloured bordered panels |
| Score summary | All 5 domain scores side by side |
| Orchestrator panel | Final holistic score + bar + combined feedback |

---

## 11. Dependencies

```bash
pip install langgraph google-generativeai python-dotenv rich
```

| Package | Purpose |
|---|---|
| `langgraph` | Graph orchestration, parallel fan-out via `Send()` |
| `google-generativeai` | Gemini API client |
| `python-dotenv` | Load `GOOGLE_API_KEY` from `.env` |
| `rich` | Terminal UI (tables, panels, colour, progress) |

---

## 12. Execution Flow (End to End)

```
python magic_graph.py
        │
        ▼
1. Load .env → configure Gemini
2. Define ESSAY_TEXT + ESSAY_PROMPT as strings
3. Compile LangGraph app
4. app.invoke({"essay_text": ..., "essay_prompt": ...})
        │
        ├── route_to_agents() → Send() × 5 (parallel)
        │       ├── agent_node(agent_index=0) → Agent 1 result
        │       ├── agent_node(agent_index=1) → Agent 2 result
        │       ├── agent_node(agent_index=2) → Agent 3 result
        │       ├── agent_node(agent_index=3) → Agent 4 result
        │       └── agent_node(agent_index=4) → Agent 5 result
        │
        ├── orchestrator_node() → final score + feedback
        │
        └── display_results(state) → rich terminal output
```

---

## 13. Key Design Decisions

| Decision | Rationale |
|---|---|
| LangGraph over plain loop | Parallel execution (3–5× faster), observable DAG, built-in checkpointing |
| `operator.add` reducer | Safe accumulation of results from parallel nodes without race conditions |
| Sort by `agent_index` before orchestrator | `Send()` parallel results arrive in non-deterministic order — must sort before passing to orchestrator |
| Gemini 2.0 Flash | Free tier, fast, handles Alpaca format naturally |
| Alpaca prompt format kept | Original MAGIC repo format — Gemini follows it without needing a wrapper strip |
| Regex JSON fallback | Gemini sometimes adds text before/after JSON — regex makes it robust |

---

## 14. What to Demonstrate

Run on 3 essays to show score differentiation:

| Essay Quality | Expected Score Range |
|---|---|
| Strong (5–6 paragraphs, well-argued) | 5–6 |
| Average (3–4 paragraphs, basic argument) | 3–4 |
| Weak (1–2 paragraphs, vague) | 1–2 |

**Key talking points:**
1. Each agent only sees its own rubric — true isolation
2. Agents 1, 2, 3 share the argumentative prompt type but score *different* aspects
3. Orchestrator synthesises — not a simple average
4. Every score has a written justification — fully explainable AI
5. LangGraph makes the pipeline a visible, inspectable graph — not a black box
