# MAGIC — How to Run

## Prerequisites

### 1. Python packages (one-time)
```bash
c
```

### 2. API Key
Go to **https://aistudio.google.com/apikey** → Create API Key → copy it.

Open `.env` in the project root and paste:
```
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXX
```

---

## Running MAGIC

Always run from the `Magic Prototype\` folder:

```bash
cd "n:\Dev\Summer Intern\Magic Prototype"
```

### Basic run (Gemini, strong essay)
```bash
python magic_graph.py
```

### Choose a different essay quality
```bash
python magic_graph.py --example strong    # expected score: 5–6
python magic_graph.py --example average   # expected score: 3–4
python magic_graph.py --example weak      # expected score: 1–2
```

### Choose Gemini model
```bash
python magic_graph.py --backend gemini --model gemini-2.0-flash   # default, fastest
python magic_graph.py --backend gemini --model gemini-1.5-flash
python magic_graph.py --backend gemini --model gemini-1.5-pro     # best quality
```

### Use Ollama (local models)
```bash
# Step 1: start Ollama server (keep this terminal open)
ollama serve

# Step 2: pull the model (one-time per model, ~4–20 GB)
ollama pull llama3.1:8b      # fast, ~4.7 GB
ollama pull gemma3:12b       # balanced, ~8 GB
ollama pull gemma3:27b       # best quality, ~17 GB

# Step 3: run
python magic_graph.py --backend ollama --model llama3.1:8b
python magic_graph.py --backend ollama --model gemma3:27b --example weak
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
═══════════════ ✨ MAGIC Essay Evaluation Pipeline ✨ ════════════════

🔧 Backend: GEMINI   ⚙ Model: gemini-2.0-flash  · essay: strong

◆ Step 1–5 · Specialised Agent Scores
┌────┬──────────────┬──────────────────────────────┬────────────────────┐
│ #  │ Agent        │ Aspect                       │ Score              │
├────┼──────────────┼──────────────────────────────┼────────────────────┤
│ 1  │ 📝 Agent 1   │ Aspect 1: Task Response      │ ████████████ 5/6  │
│ 2  │ 🧠 Agent 2   │ Aspect 2: Argumentation      │ ████████░░░░ 4/6  │
│ 3  │ 📐 Agent 3   │ Aspect 3: Organisation       │ ████████████ 5/6  │
│ 4  │ 📚 Agent 4   │ Aspect 4: Vocabulary         │ ████████████ 5/6  │
│ 5  │ ✏️  Agent 5   │ Aspect 5: Grammar            │ ████████████ 5/6  │
└────┴──────────────┴──────────────────────────────┴────────────────────┘

◆ Detailed Reasoning per Agent
[coloured panel per agent with full step-by-step reasoning]

━━━━━━━━━━━━━━━━ Score Summary ━━━━━━━━━━━━━━━━━
  📝 T1 · Prompt Response    ████████████ 5/6
  🧠 T2 · Argumentation      ████████░░░░ 4/6
  ...

🎯 Step 6 · Orchestrator Final Score
FINAL HOLISTIC SCORE: 5 / 6
[Combined student feedback from all 5 agents]
```

---

## How It Works (30 seconds)

1. **5 agents run in parallel** via LangGraph `Send()` — each sees only its own rubric
2. All 5 finish → results fan-in to the **orchestrator**
3. Orchestrator synthesises a **holistic final score**
4. Rich terminal UI renders everything

**Key principle:** Agents never see each other's scores. Only the orchestrator has full visibility.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `KeyError: GOOGLE_API_KEY` | Check `.env` file exists and key is correct |
| `ModuleNotFoundError: prompts` | Run from `Magic Prototype\` folder, not a subfolder |
| Score always 0 | Agent returned malformed JSON — add `print(raw)` in `call_llm()` |
| Ollama: connection refused | Run `ollama serve` in a separate terminal first |
| Ollama: model not found | Run `ollama pull <model-name>` first |