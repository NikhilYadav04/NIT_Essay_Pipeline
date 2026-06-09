# MAGIC vs AutoScore — Architecture & Instruction Comparison
## LLaMA 3.1:8b + Gemma3:12b · 100 Essays · ASAP Dataset

---

## 1. Runs Compared

| # | Run | Model | Pipeline | Instructions | Valid | QWK | r | MAE | Exact | ±1 |
|---|-----|-------|----------|--------------|-------|-----|---|-----|-------|----|
| 1 | MAGIC 5-Agent GRE | LLaMA 3.1:8b | MAGIC | Original | 100/100 | 0.396 | 0.454 | 0.820 | 44.0% | 77.0% |
| 2 | AutoScore 2-Agent ASAP | LLaMA 3.1:8b | AutoScore | Original | 100/100 | 0.410 | 0.546 | 0.747 | 44.4% | 82.8% |
| 3 | MAGIC 5-Agent GRE | LLaMA 3.1:8b | MAGIC | Strict Calibration | 100/100 | 0.351 | 0.394 | 0.910 | 37.0% | 78.0% |
| 4 | AutoScore 2-Agent ASAP | Gemma3:12b | AutoScore | Original | 91/100 | 0.273 | 0.404 | 0.934 | 33.0% | 75.8% |

Runs 1–3: LLaMA 3.1:8b, 100 ASAP essays, local Ollama. Run 4: Gemma3:12b, 100 essays (9 excluded due to JSON parse failures).

---

## 2. What Changed in Run 3 — Strict Calibration Instructions

Run 3 used modified prompts with explicit calibration instructions added to all 3 agent system prompts (argumentative, vocabulary, grammar) and the orchestrator prompt. The goal was to reduce LLM leniency bias — the tendency to avoid giving low scores (0/1/2) and suppress high scores (5/6).

### What was added to each agent prompt:

```
IMPORTANT CALIBRATION INSTRUCTIONS:
- These are student essays (grade 7-10 level). Score relative to student writing 
  standards, NOT graduate-level expectations.
- You MUST use the full scoring range 0-6. In a typical student dataset: 
  scores 0-2 apply to roughly 20% of essays, scores 3-4 to roughly 50%, 
  scores 5-6 to roughly 30%.
- SCORE 5-6 (HIGH): Award when the essay is genuinely strong FOR STUDENT LEVEL —
  clear position, 3+ developed paragraphs, good use of examples, mostly fluent.
  A well-written student essay SHOULD receive 5 or 6. Do NOT reserve these for 
  perfection.
- SCORE 3-4 (MID): Average student work — some development but with clear 
  weaknesses. This is the default for a mediocre attempt.
- SCORE 0-2 (LOW): Award when the essay is clearly weak — blank, off-topic, 
  incoherent, unsupported, or extremely brief. Do NOT avoid low scores out of 
  politeness.
- Both extremes matter: inflating weak essays AND suppressing strong ones are 
  equally harmful to accurate scoring.
```

### What was added to the orchestrator prompt (additionally):

```
- If multiple expert graders assign low scores (1-2), your holistic score MUST 
  reflect that — do NOT inflate to 3.
- If expert graders mostly assign 4-5, your holistic score SHOULD be 5 or 6. 
  Do NOT suppress high scores out of caution.
```

### Why this approach was chosen:
- LLMs are trained via RLHF to be encouraging and avoid being "harsh" — this causes systematic score inflation at the low end and suppression at the high end
- Adding explicit score distribution targets (20%/50%/30%) gives the model a concrete calibration reference
- Symmetric anchors for BOTH extremes prevent overcorrection in either direction
- The orchestrator-level rule prevents agent-level inflation from being compounded during synthesis

---

## 3. Metric-by-Metric Comparison

### QWK (Primary Metric — Agreement with Human)

| Run | QWK | Band |
|-----|-----|------|
| MAGIC Original | 0.396 | Fair |
| AutoScore | **0.410** | Moderate |
| MAGIC Strict | 0.351 | Fair |

AutoScore leads. Strict instructions **hurt** MAGIC — QWK dropped by 0.045. The model overcorrected on the low end, pushing mid-range essays (human score 3) down to 2.

---

### MAE (Average Grade-Point Error)

| Run | MAE | Interpretation |
|-----|-----|----------------|
| MAGIC Original | 0.820 | Less than 1 grade off |
| AutoScore | **0.747** | Closest to human |
| MAGIC Strict | 0.910 | Slightly over 1 grade off |

AutoScore is the most accurate on average. Strict MAGIC worsened MAE — confirms overcorrection.

---

### ±1% (Industry Standard Threshold)

| Run | ±1% |
|-----|-----|
| MAGIC Original | 77.0% |
| AutoScore | **82.8%** |
| MAGIC Strict | 78.0% |

AutoScore wins. 83% of scores within 1 grade of human — comfortably above the 80% industry benchmark.

---

### Exact Match %

| Run | Exact % |
|-----|---------|
| MAGIC Original | 44.0% |
| AutoScore | **44.4%** |
| MAGIC Strict | 37.0% |

AutoScore and MAGIC Original are almost identical. Strict MAGIC drops to 37% — confirms the instruction changes disrupted what was already working.

---

## 4. Confusion Matrix Analysis

### Run 1 — MAGIC Original

**Key observations:**
- Human 0 essays → predicted as 2 (1), 3 (3), and 4 (11). Severe over-scoring — the weakest essays are pushed up to 4.
- Human 2 essays → predicted as 3 (2), 4 (29), and 5 (10). Massive over-scoring — the dominant cell is at 4, not 2.
- Human 3 essays → predicted as 3 (4), 4 (15), 5 (12), 6 (2). Strong upward push — most land at 4-5 instead of 3.
- Human 4 essays → predicted as 3 (1) and 5 (7). Decent high-end sensitivity but still over-shoots to 5.
- Human 6 essays → predicted as 4 (2) and 5 (1), never 6. High-end suppression at the very top.
- Score 0, 1, and 2 are essentially never predicted — the model refuses to assign low scores.
- **Pattern: Strong positive/leniency bias — predictions cluster at 4-5 regardless of the true human score.**

---

### Run 2 — AutoScore

**Key observations:**
- Human 0 essays → predicted as 1 (2) and 2 (13). Low-end inflation present but far less severe than MAGIC — at least it reaches down to 1-2 instead of 4.
- Human 2 essays → predicted as 1 (1), 2 (31), 3 (8). 31 correct at 2 — the strongest diagonal cell across all 3 runs and the best low-range accuracy.
- Human 3 essays → predicted as 2 (20), 3 (12), 4 (1). Mild under-scoring of mid-range, but 12 correct.
- Human 4 essays → predicted as 2 (1), 3 (6), 4 (1). Struggles at the higher end — pulls 4s down to 3.
- Human 6 essays → predicted as 3 (2) and 4 (1). High-end suppression — never reaches 5-6.
- Score 0, 5, and 6 are never predicted. The model compresses everything into the 1-4 band.
- **Pattern: Conservative and consistent — tight clustering around 2-3, strongest on low-mid range, weak at the high end.**

---

### Run 3 — MAGIC Strict Instructions

**Key observations:**
- Human 0 essays → predicted as 1 (1), 2 (11), 3 (2), 4 (1). The instructions DID help reach down to 1-2 (vs Run 1 which pushed these to 4), but still no exact 0.
- Human 2 essays → predicted as 2 (29), 3 (8), 4 (4). 29 correct at 2 — a big improvement over Run 1, where Human-2 essays landed at 4.
- Human 3 essays → predicted as 2 (16), 3 (5), 4 (11), 5 (1). **Overcorrection visible — 16 of the true-3 essays were pushed down to 2, leaving only 5 correct.**
- Human 4 essays → predicted as 2 (1), 3 (4), 4 (3). Pulled down — only 3 correct at 4.
- Human 6 essays → predicted as 2 (1), 4 (1), 5 (1). Reached 5 for the first time, but the spread is erratic.
- Score distribution widened toward the low end (2 is now the dominant prediction) but at the cost of mid-range accuracy.
- **Pattern: Overcorrected downward — the instructions successfully reduced high-end inflation (good for true 0-2 essays) but pushed many true-3 essays down to 2, which is what dragged QWK below Run 1.**

---

## 5. Why Strict Instructions Lowered QWK on LLaMA 3.1:8b

The instructions were not a clean failure — they had a **mixed effect**. They successfully fixed one problem and introduced another:

**What improved (the intended effect):**
- Run 1 pushed weak essays (Human 0-2) up to score 4. Run 3 correctly scored most of these down to 1-2. The low-end inflation that plagued the original MAGIC was genuinely reduced.

**What broke (the unintended side effect):**
- The model over-applied the "give low scores" signal and pushed many **true mid-range essays (Human 3) down to 2**. 16 of the Human-3 essays were predicted as 2, leaving only 5 correct.
- Because QWK penalises distance from the human score quadratically, mass-shifting 3s down to 2s costs more than the gains from fixing the 0-2 essays — so net QWK dropped from 0.396 to 0.351.

**Why an 8b model reacts this way:**

1. **Limited instruction-following capacity** — LLaMA 3.1:8b picked up the strongest signal ("use the full range / give low scores") and applied it broadly, without the nuance to distinguish "this essay deserves 1" from "this essay deserves 3".

2. **The distribution reminder was over-applied** — telling the model "20% of essays should score 0-2" nudged it to assign low scores more liberally across the board, not just to essays that warranted them.

3. **Asymmetric uptake of the symmetric anchor** — the low-score guidance was acted on more strongly than the high-score guidance. The model over-rotated toward caution.

4. **Instruction-based calibration scales with model size** — larger models (Gemini 2.5 Pro, GPT-4o) follow complex multi-constraint instructions far more reliably. On an 8b model, architectural changes (like AutoScore's chain-of-thought) are more effective than prompt changes.

**Takeaway:** The instruction approach is sound in principle but needs either (a) a larger model that can follow the nuance, or (b) softer wording that does not over-trigger the low-score response on an 8b model.

---

## 6. Which to Use — For What Purpose

### Use AutoScore when:
- **Raw accuracy is the priority** — highest QWK, best MAE, best ±1%
- The task is pass/fail or grade-assignment where a single accurate number matters
- You are working with a smaller model (≤8b parameters)
- Simpler, more interpretable pipeline is preferred

### Use MAGIC when:
- **Rich, actionable feedback is the priority** — 5 dimensional scores tell the student WHERE they lost points
- The end product is a student-facing tool (like EssayIQ) where per-dimension breakdown is valuable
- A larger model is available (12b+) — MAGIC's multi-agent parallelism benefits more from model capability
- You want explainability for a professor, evaluator, or presentation

### The Core Tradeoff:

| | AutoScore | MAGIC |
|--|-----------|-------|
| Accuracy on 8b (QWK) | **0.410** | 0.396 |
| Feedback depth | Single holistic score | **5-dimension breakdown** |
| Speed | Faster (2 sequential calls) | Slower (5 parallel + orchestrator) |
| Small-model (8b) fit | **Better (confirmed)** | Slightly behind |
| Large-model fit | Good | Hypothesised better — *not yet confirmed* |
| Student usefulness | Low | **High** |
| Research explainability | Low | **High** |

---

## 7. Key Research Finding

> **On a small model (LLaMA 3.1:8b), architecture and accuracy do not trade off cleanly against feedback richness.**
>
> AutoScore's simpler, sequential two-agent design achieves the best agreement with human scores (QWK 0.410) — its step-by-step evidence-then-score structure keeps an 8b model grounded.
>
> MAGIC's five-agent parallel design is slightly less accurate (QWK 0.396) but produces a per-dimension breakdown that AutoScore cannot — making it more useful for a student-facing product even when the holistic number is marginally less precise.
>
> The hypothesis that MAGIC's multi-agent design pulls ahead on larger models is **plausible but not yet confirmed** — the gemma3:12b MAGIC run (QWK 0.346) did not beat llama3.1:8b, so model family and quantisation matter as much as raw size. Confirming this requires testing MAGIC on a strong instruction-following model (Gemini 2.5 Pro / GPT-4o).

The central, evidence-backed contribution of this evaluation is the **accuracy-vs-interpretability tradeoff** between the two architectures on a fixed small model — not a claim about scaling, which remains an open question for the next round of runs.

---

## 8. Run 4 — AutoScore + Gemma3:12b

**QWK: 0.273 | r: 0.404 | MAE: 0.934 | Exact: 33.0% | ±1: 75.8% | Valid: 91/100**

### Overview

This run tested whether a larger model (Gemma3:12b) improves AutoScore's accuracy beyond LLaMA 3.1:8b (QWK 0.410). The result was the opposite — Gemma3:12b performed worse on every metric. This is a significant finding because it demonstrates that **model size does not guarantee better performance in structured pipelines** — instruction-following reliability matters more.

9 of 100 essays failed with a JSON parse error: Gemma3:12b wrapped its output in markdown code fences (` ```json ... ``` `) despite the scoring prompt explicitly instructing `"Output ONLY the JSON object — no preamble, no explanation, no markdown code fences"`. LLaMA 3.1:8b had 0 such failures on the same pipeline.

### Confusion Matrix Analysis

- Human 0 essays → predicted as 1 (7) and 2 (8). Never predicts 0 — low-end inflation present but score range is lower than MAGIC's 4-5.
- Human 2 essays → predicted as 1 (11), 2 (28), 3 (1). 28 correct — reasonable low-range accuracy but 11 under-scored to 1.
- Human 3 essays → predicted as 1 (6), 2 (18), 3 (2). Only 2 correct — severe under-scoring of mid-range, pulled to 1-2.
- Human 4 essays → predicted as 2 (6) and 3 (2). Never reaches 4 — high-end pulled down.
- Human 6 essays → predicted as 3 (2) only. High-end suppression.
- Scores 4, 5, 6 almost never predicted.
- **Pattern: Strong under-scoring bias — predictions compressed into the 1-2 band. More conservative than LLaMA AutoScore which clustered at 2-3.**

### Why Gemma3:12b Underperforms LLaMA 3.1:8b on AutoScore

AutoScore's architecture depends on two strict JSON contracts:
1. SRCE Agent must output a clean evidence JSON (no scores, no extra text)
2. Scoring Agent must output a clean scoring JSON (no markdown, no preamble)

Gemma3:12b fails the second contract reliably — it adds markdown fences even when explicitly told not to. This produces 9% parse failures (essays scored as -1, excluded). The valid 91 essays are also affected: the model's tendency to add commentary around JSON suggests its internal representation is less strictly "JSON-first," which may degrade the scoring quality even when parsing succeeds.

LLaMA 3.1:8b follows the JSON-only instruction faithfully (0 failures), making it more compatible with AutoScore's structured pipeline despite being a smaller model.

**This is the key finding: AutoScore's accuracy depends on instruction-following precision, not raw model scale. A model that reliably follows the JSON contract outperforms a larger model that doesn't.**

---

## 9. Cross-Run Comparison — All 4 Runs

| Rank | Run | Model | QWK | Pattern |
|------|-----|-------|-----|---------|
| 1 | AutoScore | LLaMA 3.1:8b | **0.410** | Conservative, strong low-mid range |
| 2 | MAGIC Original | LLaMA 3.1:8b | 0.396 | Leniency bias, over-scores to 4-5 |
| 3 | MAGIC Strict | LLaMA 3.1:8b | 0.351 | Overcorrected, pushes 3s → 2s |
| 4 | AutoScore | Gemma3:12b | 0.273 | Under-scoring bias, JSON failures |

**Key insight:** LLaMA 3.1:8b is the best-performing model across both architectures. Its reliability on JSON contracts (AutoScore) and consistent scoring behaviour (MAGIC) outweigh Gemma3:12b's parameter advantage. This strongly suggests **instruction-following accuracy is the primary quality driver** for structured multi-agent pipelines, not model size.

---

## 10. Next Steps

1. **Run MAGIC with `--rubric asap`** on LLaMA 3.1:8b — true apples-to-apples against AutoScore (both ASAP-native)
2. **Test MAGIC on Gemini 2.5 Pro** — larger model where multi-agent architecture should shine, and which follows JSON instructions reliably
3. **Fix Gemma3:12b JSON compliance** — add a post-processing strip of markdown fences before parsing, retry failed essays
4. **Score calibration post-processing** — apply a learned offset to correct systematic bias rather than relying on prompt instructions
5. **Hybrid approach** — AutoScore for holistic accuracy + MAGIC for per-dimension feedback, combined in one pipeline
