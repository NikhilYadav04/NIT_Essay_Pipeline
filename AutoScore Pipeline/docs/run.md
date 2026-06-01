# AutoSCORE — How to Run

## Prerequisites

### 1. Python packages (one-time)
```bash
pip install langgraph google-generativeai python-dotenv rich requests
```

### 2. API Key
Go to **https://aistudio.google.com/apikey** → Create API Key → copy it.

Open `.env` in the project root and paste:
```
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXX
```

---

## Running AutoSCORE

Always run from the `AutoScore Pipeline\` folder:

```bash
cd "n:\Dev\Summer Intern\AutoScore Pipeline"
```

### Basic run (Gemini, strong essay)
```bash
python autoscore_graph.py
```

### Choose a different essay quality
```bash
python autoscore_graph.py --example strong    # expected score: 5–6
python autoscore_graph.py --example average   # expected score: 3–4
python autoscore_graph.py --example weak      # expected score: 1–2
```

### Choose Gemini model
```bash
python autoscore_graph.py --backend gemini --model gemini-2.0-flash   # default, fastest
python autoscore_graph.py --backend gemini --model gemini-1.5-flash
python autoscore_graph.py --backend gemini --model gemini-1.5-pro     # best quality
```

### Use Ollama (local models)
```bash
# Step 1: start Ollama server (keep this terminal open)
ollama serve

# Step 2: pull the model (one-time per model)
ollama pull llama3.1:8b      # fast, ~4.7 GB
ollama pull gemma3:12b       # balanced, ~8 GB
ollama pull gemma3:27b       # best quality, ~17 GB

# Step 3: run
python autoscore_graph.py --backend ollama --model llama3.1:8b
```

### Full flag reference
```
--backend   gemini | ollama          (default: gemini)
--model     any model name           (default: gemini-2.0-flash / llama3.1:8b)
--example   strong | average | weak  (default: strong)
```

---

## What the Output Shows

```
═════════════ 🔬 AutoSCORE — Two-Agent Essay Evaluation ═════════════

🔧 Backend: GEMINI   ⚙ Model: gemini-2.0-flash  · essay: strong

──────── 📋 Step 1 — Agent 1 (SRCE): Structured Evidence ────────
Agent 1 read the rubric and essay. It extracted evidence only — no scores assigned.

┌─ Evidence JSON — Agent 1 Output ─────────────────────────────────┐
│ {                                                                  │
│   "trait_1_task_response": {                                      │
│     "addressed": true,                                            │
│     "has_clear_position": true,                                   │
│     "position_quote": "The argument...fails to substantiate...",  │
│     "addresses_prompt_directly": true,                            │
│     "notes": "Essay identifies causation and transferability..."  │
│   },                                                              │
│   "trait_2_argument_quality": { ... },                           │
│   ...                                                             │
│ }                                                                  │
└────────────────────────────────────────────────────────────────────┘

──────── ⚖️ Step 2 — Agent 2 (Scoring): Rubric Judgment Applied ────────
Agent 2 used the evidence JSON as its primary source.

┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Trait               │ Score                │ Evidence Used        │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ 📝 Task Response     │ ████████████  5/6   │ has_clear_position.. │
│ 🧠 Argument Quality  │ ████████████  5/6   │ example_count: 4...  │
│ 📐 Organisation      │ ████████░░░░  4/6   │ has_transitions...   │
│ 📚 Language & Style  │ ████████████  5/6   │ vocabulary_range...  │
│ ✏️  Grammar           │ ████████████  6/6   │ error_frequency...   │
└─────────────────────┴──────────────────────┴──────────────────────┘

[Per-trait panels with rubric match + evidence used + student feedback]

🎯 Final Holistic Score
HOLISTIC SCORE: 5 / 6
[Reasoning + combined student feedback]

── Audit Trail ──
Agent 1 extracted evidence → Agent 2 scored from evidence → Score above
Every score is traceable to a specific field in the Evidence JSON above.
```

---

## How It Works (30 seconds)

1. **Agent 1 (SRCE)** reads the essay + rubric → outputs a **structured JSON** of evidence
   - No scores. Evidence extraction only.
2. **Agent 2 (Scoring)** reads the JSON + rubric → assigns scores **from the evidence**
   - It is anchored to the JSON — it cannot freely re-read the essay and score holistically
3. Rich terminal UI shows the full audit trail: evidence → judgment → score

**Key principle:** Every score is traceable to a specific JSON field. This is the auditability guarantee.

---

## Why the JSON Intermediate Matters

> If Agent 2 ignored the JSON and just read the essay — you'd have a single-agent system.
> The JSON is the entire architectural point of AutoSCORE.

The `evidence_used` column in the score table shows **exactly which JSON field** drove each score.
This makes every grade fully explainable and auditable.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `KeyError: GOOGLE_API_KEY` | Check `.env` file exists and key is correct |
| `ModuleNotFoundError` | Run from `AutoScore Pipeline\` folder, not a subfolder |
| Evidence JSON is `{}` | Agent 1 returned malformed JSON — check raw output |
| All scores 0 | Evidence dict empty — Agent 1 failed silently |
| Ollama: connection refused | Run `ollama serve` in a separate terminal first |
| Ollama: model not found | Run `ollama pull <model-name>` first |
| JSON parse error | Model wrapped JSON in ``` fences — `strip_markdown_fences()` handles this |
