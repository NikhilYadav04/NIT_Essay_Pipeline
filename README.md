# Automated Essay Scoring Research

## Overview

This repository contains the full research implementation produced during the summer internship. The project investigates whether multi-agent LLM architectures can produce accurate, interpretable, and rubric-aligned essay scores comparable to human graders.

Two pipeline architectures are implemented and benchmarked:

| Pipeline | Architecture | Scoring Strategy |
|---|---|---|
| **MAGIC Pipeline** | Multi-Agent Graph with 5 specialist agents + 1 orchestrator | Parallel per-dimension scoring → holistic synthesis |
| **AutoScore Pipeline** | Single LangGraph agent with structured rubric prompting | End-to-end single-pass scoring |

---

## Repository Structure

```
NIT_Essay_Pipeline/
├── Magic Pipeline/          # Multi-Agent Intelligent Grading & Critique (MAGIC)
├── AutoScore Pipeline/      # Single-agent AutoScore baseline
├── Batch Evaluation/        # Automated benchmarking and metric analysis
├── Datasets/                # Essay datasets used for evaluation
└── Docs Research/           # Research papers and project documentation
```

---

## Magic Pipeline

**Architecture:** Multi-Agent LangGraph system implementing the MAGIC (Multi-Agent Intelligent Grading and Critique) framework.

### How It Works

```
Essay Input
     │
     ▼
┌─────────────────────────────────────────────────────┐
│              Parallel Agent Fan-Out                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Agent 1  │ │ Agent 2  │ │ Agent 3  │  ...×5     │
│  │ Task     │ │ Argument │ │ Organis- │            │
│  │ Response │ │ Quality  │ │ ation    │            │
│  └──────────┘ └──────────┘ └──────────┘            │
└──────────────────────┬──────────────────────────────┘
                       │ (5 scores + feedback)
                       ▼
              ┌─────────────────┐
              │  Orchestrator   │
              │  Agent          │
              │  (holistic      │
              │   synthesis)    │
              └────────┬────────┘
                       │
                       ▼
              Final Score (0–6) + Holistic Feedback
```

### 5 Evaluation Dimensions (GRE Analytical Writing Rubric)

| Agent | Dimension | What It Evaluates |
|---|---|---|
| Agent 1 | Task Response | How completely the prompt is addressed |
| Agent 2 | Argument Quality | Reasoning, evidence, and complexity of ideas |
| Agent 3 | Organisation | Paragraph structure, coherence, transitions |
| Agent 4 | Vocabulary & Style | Lexical richness and sentence variety |
| Agent 5 | Grammar & Mechanics | Syntactic accuracy and punctuation |

### Key Files

| File | Purpose |
|---|---|
| `magic_graph.py` | LangGraph graph definition — nodes, edges, parallel Send() routing |
| `llm_client.py` | LLM backend abstraction (supports Ollama local + Google Gemini) |
| `prompts/agents.py` | Per-agent system prompts and GRE rubric definitions |
| `prompts/orchestrator.py` | Orchestrator synthesis prompt |
| `display.py` | CLI result display and formatted score output |

### Supported LLM Backends

```ini
# Magic Pipeline/.env
MAGIC_BACKEND=ollama          # or: gemini
MAGIC_MODEL=llama3.1:8b       # or: gemini-2.0-flash
GOOGLE_API_KEY=your_key_here  # required for Gemini backend only
```

---

## AutoScore Pipeline

**Architecture:** Single LangGraph agent implementing the AutoScore methodology — a structured single-pass scorer using detailed rubric-conditioned prompting.

### How It Works

```
Essay Input + Rubric
        │
        ▼
┌───────────────────┐
│  AutoScore Agent  │  ← Single LLM call with full rubric context
│  (structured      │
│   rubric prompt)  │
└────────┬──────────┘
         │
         ▼
  Per-Dimension Scores + Feedback
```

Unlike MAGIC's parallel agents, AutoScore performs a single holistic evaluation pass using a comprehensive rubric prompt, serving as the **baseline** for comparison.

### Key Files

| File | Purpose |
|---|---|
| `autoscore_graph.py` | LangGraph graph definition for single-agent scoring |
| `prompts/scoring_prompt.py` | Rubric-conditioned scoring prompt template |
| `display.py` | Result display utilities |

---

## Batch Evaluation

Automated benchmarking suite for running both pipelines at scale and computing agreement metrics against human gold-standard scores.

### Pipeline Support

Both MAGIC and AutoScore pipelines can be benchmarked using the same runner:

```bash
# Run MAGIC pipeline
python batch_runner.py --backend gemini --model gemini-2.0-flash --pipeline magic

# Run AutoScore pipeline  
python batch_runner.py --backend ollama --model llama3.1:8b --pipeline autoscore
```

### Key Files

| File | Purpose |
|---|---|
| `batch_runner.py` | Main benchmarking runner — iterates dataset, calls pipeline, logs results |
| `analyze.py` | Metric computation: QWK, Pearson r, MSE, score distributions |
| `prepare_data.py` | Dataset preprocessing and formatting for pipeline input |
| `results/` | Output CSVs, metric summaries, and comparison reports |

### Metrics Computed

| Metric | Description |
|---|---|
| **QWK** | Quadratic Weighted Kappa — primary AES agreement metric |
| **Pearson r** | Linear correlation with human scores |
| **MSE / RMSE** | Mean squared error against gold labels |
| **Score Distribution** | Histogram of predicted vs. human scores |

---

## Docs Research

Reference papers and project documentation used throughout the research.

| Document | Description |
|---|---|
| `MAGIC_Paper.pdf` | Original MAGIC multi-agent AES paper (primary reference) |
| `AutoScore_Paper.pdf` | AutoScore methodology paper (baseline reference) |
| `LLM_Strict_Grading_Research.pdf` | Research on strict rubric-following in LLM graders |
| `Research_Survey.pdf` | Broader AES literature survey |
| `MAGIC_Pipeline_AES_Report.docx` | Project implementation report |
| `MAGIC_vs_AutoScore_Comparison.docx` | Comparative analysis of both architectures |
| `Essay_Evaluation_Pipeline_Overview.docx` | High-level pipeline design documentation |

---

## Datasets

Essay datasets used to benchmark both pipelines against human gold-standard scores.

| File | Essays | Source | Prompt Type |
|---|---|---|---|
| `ASAP_100.csv` | 100 | ASAP (Automated Student Assessment Prize) | Short student essays, multiple prompts |
| `GRE_100.csv` | 100 | GRE Analytical Writing dataset | Long-form argumentative essays |

Both datasets contain:
- **Essay text** — the full student essay
- **Human score** — gold-standard score assigned by trained human graders (0–6 scale)
- **Prompt** — the essay question/task the student responded to

These are used by the **Batch Evaluation** pipeline to compute QWK and other agreement metrics between LLM-predicted scores and human scores.

---

## Technology Stack

| Component | Technology |
|---|---|
| Graph Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Local LLM Runtime | [Ollama](https://ollama.com) |
| Cloud LLM | Google Gemini API |
| Language | Python 3.10+ |
| Key Libraries | `langchain`, `google-genai`, `pandas`, `scikit-learn` |

---

## Research Questions

1. Does parallel multi-agent scoring (MAGIC) produce more consistent and rubric-aligned scores than single-agent scoring (AutoScore)?
2. How do open-source local models (Llama 3.1) compare to commercial APIs (Gemini) on this task?
3. What is the QWK agreement between LLM-based graders and human annotators on GRE-style essay prompts?
