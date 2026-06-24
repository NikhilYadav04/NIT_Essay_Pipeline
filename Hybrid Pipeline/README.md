# Hybrid Pipeline — MAGIC + AutoSCORE

> Evidence-Anchored Multi-Agent Essay Evaluation

---

## What This Is

A third essay evaluation prototype that combines both architectures:

- **AutoSCORE's SRCE Agent** runs first and extracts a structured JSON evidence map from the essay
- **MAGIC's 5 specialist agents** then each receive that evidence JSON alongside the essay before scoring
- **MAGIC's Orchestrator** synthesises all 5 scores into a final holistic score

**Why this should outperform either alone:**

| Pipeline | QWK | Feedback |
|---|---|---|
| AutoSCORE | 0.410 | Thin (holistic only) |
| MAGIC | 0.396 | Rich (per-trait) |
| **Hybrid (predicted)** | **> 0.410** | **Rich (per-trait, evidence-anchored)** |

---

## Pipeline Flow

```
Essay input
    │
    ▼
[Stage 1] SRCE Agent (AutoSCORE)
    Reads essay + full 5-trait rubric
    Outputs structured JSON evidence record (no scores, just facts)
    │
    ▼
[Stage 2] 5 MAGIC Agents — run in PARALLEL
    Each agent receives:
      - Their own individual rubric (e.g. Agent 1 → Task Response rubric only)
      - The SRCE evidence JSON from Stage 1
      - The original essay (for reference only)
    Each outputs: score (0–6) + detailed examiner comment
    │
    ▼
[Stage 3] Orchestrator (MAGIC)
    Reads all 5 scores + comments
    Outputs: final holistic score + consolidated feedback

Total LLM calls per essay: 7  (1 SRCE + 5 agents + 1 orchestrator)
```

---

## Setup

### 1. Copy your `.env` file into this folder

```
Hybrid Pipeline/.env
```

It must contain:
```
GOOGLE_API_KEY=your_key_here
```

### 2. Install dependencies

```bash
pip install langgraph langchain google-generativeai python-dotenv rich requests
```

> These are the same packages already used by Magic Pipeline and AutoScore Pipeline.
> If those work, this will too — no new installs needed.

---

## How to Run

### Basic run (Gemini, strong essay example)
```bash
cd "Hybrid Pipeline"
python hybrid_graph.py
```

### Choose an essay quality
```bash
python hybrid_graph.py --example strong    # Well-argued, 5 paragraphs (default)
python hybrid_graph.py --example average   # Decent, 4 paragraphs
python hybrid_graph.py --example weak      # Poor, 2 paragraphs
```

### Use Gemini with a specific model
```bash
python hybrid_graph.py --backend gemini --model gemini-2.0-flash
python hybrid_graph.py --backend gemini --model gemini-1.5-pro
```

### Use Ollama (local model, no API key needed)
```bash
# Make sure Ollama is running first:
ollama serve

# Then run:
python hybrid_graph.py --backend ollama --model llama3.1:8b
python hybrid_graph.py --backend ollama --model gemma3:12b
python hybrid_graph.py --backend ollama --model gemma3:27b
```

### Combine flags
```bash
python hybrid_graph.py --backend ollama --model llama3.1:8b --example weak
```

---

## 3-Way Ablation Study (Recommended)

Run all three pipelines on the same essay and model to compare scores:

```bash
# Terminal 1 — MAGIC
cd "Magic Pipeline"
python magic_graph.py --example strong

# Terminal 2 — AutoSCORE
cd "AutoScore Pipeline"
python autoscore_graph.py --example strong

# Terminal 3 — Hybrid
cd "Hybrid Pipeline"
python hybrid_graph.py --example strong
```

Record the three final scores side-by-side and compare against the ASAP human score.
This is your ablation study — same essay, same model, measurable difference.

---

## Output

The terminal display shows three sections in order:

1. **SRCE Evidence Record** — the JSON extracted before any agent scored
2. **5 Agent Scores** — table + per-agent detailed feedback
3. **Final Orchestrator Score** — holistic score + consolidated feedback + pipeline audit line

---

## File Structure

```
Hybrid Pipeline/
├── prompts/
│   ├── __init__.py
│   ├── base.py              ← copied from Magic Pipeline (unchanged)
│   ├── orchestrator.py      ← copied from Magic Pipeline (unchanged)
│   ├── srce_prompt.py       ← copied from AutoScore Pipeline (unchanged)
│   └── agents.py            ← MODIFIED: evidence_json injected into all 3 prompts
├── hybrid_graph.py          ← main pipeline (LangGraph, 3-stage)
├── display.py               ← Rich terminal display
├── llm_client.py            ← merged LLM client (Gemini + Ollama)
├── .env                     ← your API key (copy from Magic Pipeline)
└── README.md                ← this file
```

---

## What Changed vs. MAGIC

The only code change from MAGIC is in `prompts/agents.py`:

- `format_prompt_inference()` accepts a new 4th parameter: `evidence_json: str = ""`
- When provided, each agent's prompt includes a `<evidence_record>` block with the SRCE JSON
- Each agent is instructed to use that record as their **primary factual foundation** instead of re-reading the essay freely

Everything else — rubrics, orchestrator, output format, CLI flags — is identical to MAGIC.
