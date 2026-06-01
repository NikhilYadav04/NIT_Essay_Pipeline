# MAGIC Pipeline — Automated Essay Scoring Analysis Report

---

## 1. Title

**Evaluating MAGIC Multi-Agent Pipeline for Automated Essay Scoring on the ASAP Dataset**
*A comparative study of model selection (LLaMA 3.1:8b vs Mistral:7b) and rubric design (5-agent GRE vs 3-agent ASAP) on 100 student essays*

---

## 2. Key Terms Explained

### Graph Terms

**QWK — Quadratic Weighted Kappa**
The primary metric in essay scoring research. Measures agreement between AI and human examiner. Errors of 2+ grades are penalised much more heavily than errors of 1 grade.
- 0.00–0.20 → Poor
- 0.20–0.40 → Fair
- 0.40–0.60 → Moderate
- 0.60–0.80 → Substantial
- 0.80–1.00 → Excellent

**r — Pearson Correlation Coefficient**
Measures whether AI scores move in the same direction as human scores. When humans give high, does AI give high? Range: -1 to +1. Higher is better.

**MAE — Mean Absolute Error**
Average number of grade points the AI is off from the human score. On a 0–6 scale, MAE below 1.0 is considered good. Lower is better.

**Exact %**
Percentage of essays where the AI score exactly matched the human score.

**±1 %**
Percentage of essays where the AI score was within 1 grade point of the human. This is the industry-standard acceptable threshold for automated essay scoring — even human raters don't always agree exactly.

### Confusion Matrix Terms
- **Rows** = Human examiner's score
- **Columns** = AI predicted score
- **Diagonal numbers** = correct predictions (AI matched human exactly)
- **Off-diagonal numbers** = errors
- Numbers **above diagonal** = AI under-scored
- Numbers **below diagonal** = AI over-scored

### Scatter Plot
- Each dot = one essay
- **Dashed diagonal line** = perfect agreement zone
- Dots close to the line = accurate predictions
- Dots clustered above the line = AI under-scoring
- Dots clustered below the line = AI over-scoring

---

## 3. Complete Analysis of All 4 Runs

---

### Run 1 — MAGIC 3-Agent ASAP Rubric + LLaMA 3.1:8b
**QWK: 0.220 | r: 0.260 | MAE: 1.110 | Exact: 29% | ±1: 69%**

**Overview:**
This run tested whether simplifying the rubric from 5 GRE-style agents to 3 student-focused agents would help LLaMA better align with human scores on the ASAP dataset. The 3 agents evaluated Content & Task Response, Organisation & Development, and Language & Conventions respectively — dimensions more appropriate for middle and high school persuasive writing than the original GRE criteria.

**Scatter Plot Analysis:**
Predictions are spread across scores 2–4, which shows the model is at least using the scoring range rather than collapsing to a single value. The diagonal alignment is present but weak — many dots are scattered above the dashed line, indicating a consistent pattern of under-scoring. A notable outlier is visible where a score 0 essay was predicted as 3, and a score 6 essay was predicted as 4, suggesting the model struggles at the extremes of the scale. The overall spread suggests the model has some sensitivity to essay quality but lacks the precision to place scores accurately.

**Confusion Matrix Analysis:**
The brightest cell is at human=2, predicted=2 with 18 correct predictions — the model performs best on low-scoring essays. However, the off-diagonal pattern reveals a clear under-scoring trend: 13 essays where the human gave a score of 3 were predicted as 2, and 9 essays where the human gave 2 were predicted as 3, showing instability in the 2–3 range. Score 0 essays were predicted as 2 or 3 in most cases, meaning the model cannot reliably identify very weak or off-topic essays. Score 4 essays were largely predicted as 3, continuing the under-scoring pattern. Only 3 essays at score 4 were correctly identified.

**Why This Happened:**
The simplified 3-agent rubric gave the model less structured criteria to anchor scores against. With fewer and broader rubric dimensions, the model had more ambiguity in deciding where an essay sits on the 0–6 scale. Paradoxically, more detailed rubrics help smaller models score more consistently.

**Assessment:** Fair agreement. The rubric simplification did not achieve the intended improvement — it introduced more score uncertainty rather than reducing domain mismatch.

---

### Run 2 — MAGIC 3-Agent ASAP Rubric + Mistral:7b
**QWK: 0.157 | r: 0.394 | MAE: 2.020 | Exact: 7% | ±1: 29%**

**Overview:**
This run applied the same 3-agent ASAP rubric to Mistral:7b to test whether the rubric simplification would help a different model. Mistral:7b is a 7 billion parameter model from Mistral AI, comparable in size to LLaMA 3.1:8b but with different training characteristics. The results reveal a fundamental model-level scoring bias that no rubric change can resolve.

**Scatter Plot Analysis:**
The scatter plot shows a severe clustering of predictions in the 4–5 range regardless of what human scores were assigned. Essays that humans scored as 0, 1, 2, and 3 are all being predicted at 4 or 5 — a near-complete failure to distinguish between weak and average essays. The Pearson r of 0.394 appears misleadingly decent; the directional correlation exists because some higher-scoring essays also happen to get higher AI scores, but the absolute placement is consistently wrong by 2+ grades for most essays. There is one visible case of a score 6 essay being correctly predicted at 6, but this appears to be coincidence rather than a reliable pattern.

**Confusion Matrix Analysis:**
Column 4 completely dominates the matrix — 23 essays where the human gave score 2 were predicted as 4, and 10 essays where the human gave score 3 were predicted as 4. This means the model is systematically over-scoring average and below-average essays by 2 full grade points. Score 0 essays were predicted as 4 or 5 in 12 out of 15 cases — the model is giving near-maximum scores to essays that humans rated as off-topic or incomprehensible. Even the 3-agent ASAP rubric, which explicitly describes score 0 as "off-topic, blank, or incomprehensible," could not override Mistral's tendency to score generously. The diagonal has almost no activity — only 7% exact match confirms that correct predictions are extremely rare.

**Why This Happened:**
Mistral:7b appears to have a strong prior toward producing mid-to-high scores when asked to grade essays. This is likely a result of its instruction fine-tuning, which may have trained the model to be encouraging rather than critical. This behaviour is a model-level characteristic that cannot be corrected through rubric design alone — it would require either a different model or explicit score calibration post-processing.

**Assessment:** Poor agreement. The ASAP rubric made no meaningful difference to Mistral's scoring behaviour. The core issue is a model-level high-score bias.

---

### Run 3 — MAGIC 5-Agent GRE Rubric + LLaMA 3.1:8b *(Best Result)*
**QWK: 0.396 | r: 0.454 | MAE: 0.820 | Exact: 44% | ±1: 77%**

**Overview:**
This is the best performing configuration in the entire study. The original 5-agent GRE rubric assigns one specialist agent each to: (1) Quality of response to the prompt, (2) Considering complexity of the issue, (3) Organisation and development, (4) Vocabulary and sentence variety, and (5) Grammar and mechanics. Each agent evaluates one specific dimension and passes its score and reasoning to the orchestrator, which synthesises a final holistic score. Despite these rubrics being designed for graduate-level GRE analytical writing, they provided sufficient structure for LLaMA to produce scores meaningfully aligned with human judgement on student essays.

**Scatter Plot Analysis:**
This scatter plot shows the strongest diagonal alignment of all four runs. Predictions are spread across scores 2–4, broadly tracking the human score distribution. The concentration of dots near the 2–3 range reflects the actual ASAP dataset distribution — most student essays genuinely fall in the lower-to-middle range. The under-scoring bias is present but predictable: the model consistently scores 1 grade below the human in many cases, which is a systematic and correctable pattern. There are no extreme outliers — no essays with human score 0 being predicted at 5 or 6, which shows the model has some ability to identify weak writing.

**Confusion Matrix Analysis:**
The strongest diagonal performance of all 4 runs. 29 correct predictions at score 2, 11 correct at score 3, and 4 correct at score 4 represent genuine agreement between AI and human. The main error pattern is under-scoring in the 2–3 range: 13 essays where the human gave 3 were predicted as 2, and 2 essays where the human gave 3 were predicted as 2 on the lower end. Score 0 essays are mostly predicted at score 2 — the model cannot identify very weak essays but at least doesn't over-score them dramatically. Score 4 essays show some agreement (4 correct) with spillover into score 3 (3 essays). The confusion matrix is compact and close to the diagonal compared to all other runs, confirming this as the best-calibrated configuration.

**Why This Performed Best:**
The 5 detailed GRE rubric dimensions gave LLaMA clear, structured scoring criteria across multiple aspects of writing. Even though these criteria were designed for GRE, the underlying qualities they measure — argumentation, organisation, vocabulary, grammar — are universal to good writing at any level. The detailed per-aspect scoring gave the orchestrator more signal to work with, resulting in a more calibrated final score. LLaMA 3.1:8b also appears to be better suited to following structured rubric instructions than Mistral:7b.

**Assessment:** Fair to borderline Moderate agreement. MAE of 0.82 means on average the model is off by less than 1 grade — practically useful. 77% of essays within ±1 of human meets the industry acceptable threshold. This is the recommended configuration for continued development.

---

### Run 4 — MAGIC 5-Agent GRE Rubric + Mistral:7b
**QWK: 0.108 | r: 0.355 | MAE: 2.030 | Exact: 4% | ±1: 30%**

**Overview:**
This run used the full original MAGIC pipeline as designed — 5 GRE specialist agents feeding into an orchestrator — but with Mistral:7b as the underlying model. This configuration represents the most direct test of model quality, as the rubric and pipeline architecture are unchanged from the original design. The results confirm that Mistral:7b is fundamentally unsuited to this scoring task regardless of rubric quality.

**Scatter Plot Analysis:**
The scatter plot reveals the most severe scoring bias of all four runs. Almost every prediction falls between 4 and 5, regardless of the human score. Essays that humans scored as 0, 1, 2, 3, and even 4 are all being predicted in the same narrow 4–5 band. The dashed diagonal line has almost no dots near it below score 4. The Pearson r of 0.355 suggests a weak correlation exists — some higher-scoring essays do receive slightly higher predictions — but the absolute score placement is consistently wrong. This is the clearest visual demonstration of a model with a fixed high-score prior that the rubric cannot override.

**Confusion Matrix Analysis:**
The confusion matrix shows the most extreme off-diagonal pattern of all 4 runs. Column 4 and 5 absorb the vast majority of predictions. In the most striking cases: essays with human score 0 were predicted as 4 or 5 in 12 of 15 cases; essays with human score 2 were predicted as 4 in 29 cases out of 41 — that is 71% of all score-2 essays being over-scored by 2 full grade points. The diagonal has almost no activity with only 4% exact match being the lowest of all runs. Score 3 essays show some spillover into the 4–5 range, and even score 4 essays are sometimes predicted at 5. The orchestrator node, which synthesises the 5 agent scores, is unable to moderate the consistently inflated individual agent scores because all 5 agents share the same Mistral model and therefore the same high-score bias.

**Why This Performed Worst:**
The 5 detailed rubric agents each call the same underlying Mistral model, which applies a consistent high-score interpretation to every rubric dimension. When all 5 agents over-score, the orchestrator receives 5 inflated scores and naturally produces an inflated holistic score. The multi-agent architecture actually amplifies the bias in this case — more agents means more opportunities for the same bias to compound. This is a fundamental incompatibility between Mistral:7b's scoring behaviour and the MAGIC pipeline's rubric-driven approach.

**Assessment:** Poor agreement. This is the lowest performing configuration in the study. The combination of the most detailed rubric with the model most prone to bias produced the worst results, demonstrating that rubric quality alone cannot compensate for model-level scoring tendencies.

---

## 4. Dataset Information

**Dataset: ASAP (Automated Student Assessment Prize)**
- Originally released by Kaggle in partnership with the Hewlett Foundation
- Contains real student persuasive essays written by middle and high school students (grades 7–10)
- Essays written in response to specific persuasive writing prompts on topics such as technology, community, and social issues
- Human scores originally on a 1–5 scale, **normalised to 0–6** for this study to match the MAGIC pipeline's native scoring range
- **100 essays** used for this evaluation — a representative subset of the full ASAP dataset
- Score distribution is skewed toward the lower-middle range (scores 2–3), which is typical for student writing populations
- Essays vary significantly in length, vocabulary, and argument sophistication, providing a genuine test of scoring discrimination ability

**Important context:** The MAGIC pipeline was originally designed and prompted for GRE analytical writing — graduate-level argumentative essays requiring complex argumentation and sophisticated language. ASAP essays are shorter, simpler student writing with different quality indicators. This domain mismatch was a central hypothesis tested in this study through the rubric variation experiment (Runs 1 and 2 vs Runs 3 and 4).

---

## 5. Final Verdict

### Overall Rankings

| Rank | Configuration | QWK | MAE | ±1% | Grade |
|------|--------------|-----|-----|-----|-------|
| 1 | LLaMA 3.1:8b + GRE 5-agent | **0.396** | **0.820** | **77%** | Fair–Moderate |
| 2 | LLaMA 3.1:8b + ASAP 3-agent | 0.220 | 1.110 | 69% | Fair |
| 3 | Mistral:7b + ASAP 3-agent | 0.157 | 2.020 | 29% | Poor |
| 4 | Mistral:7b + GRE 5-agent | 0.108 | 2.030 | 30% | Poor |

---

### Key Conclusions

**1. Model choice matters more than rubric design**
The gap between LLaMA (0.396) and Mistral (0.108–0.157) on the same rubric is far larger than the gap caused by changing the rubric. Model selection is the primary lever for performance improvement in this pipeline.

**2. The rubric reduction strategy did not help**
Reducing from 5 GRE agents to 3 ASAP agents actually hurt LLaMA's performance (0.396 → 0.220). The GRE rubric, despite being designed for graduate writing, provides more structured and specific scoring criteria that helps the model anchor its scores. Simpler, broader rubrics introduce more ambiguity and give smaller models less guidance on where to place scores.

**3. Mistral has an inherent high-score bias**
Across both rubrics, Mistral consistently over-scores essays by 2+ grades. This is not a rubric problem — it is a model behaviour characteristic where Mistral interprets grading prompts generously. This bias cannot be fixed by rubric design changes alone.

**4. LLaMA has a consistent under-scoring bias**
LLaMA consistently predicts 1 grade lower than human scores, particularly in the 2–3 range. Importantly, this is a predictable and systematic bias — unlike Mistral's erratic over-scoring. A simple post-processing score calibration step (adding +1 to all predictions) could potentially improve LLaMA's QWK further without any model changes.

**5. Reducing sub-agents strategy — findings**
The hypothesis that reducing from 5 GRE agents to 3 ASAP-specific agents would better match student essay evaluation was not supported by the data. The 5-agent architecture outperformed the 3-agent architecture for LLaMA by a margin of 0.176 QWK points. The reason appears to be that more granular rubric criteria give the model more decision points to reason through, ultimately producing a more calibrated holistic score. For future work, the recommended approach would be to keep the 5-agent architecture but rewrite the rubric content to be ASAP-appropriate, rather than reducing the number of agents.

**6. Best result is still "Fair" — not production-ready on local models**
QWK of 0.396 is below the 0.60 threshold considered acceptable for real deployment in educational settings. This is expected given the use of 7–8 billion parameter local models running without GPU acceleration. Research-grade AES systems typically use models with 70B+ parameters or cloud-based frontier models. With a stronger backend such as Gemini or GPT-4, the same MAGIC pipeline architecture would likely achieve Substantial agreement (QWK 0.60+).

---

### What This Study Demonstrates

This evaluation successfully demonstrates that the MAGIC multi-agent pipeline architecture is a sound and promising approach to automated essay scoring. The pipeline produces structured, explainable scores across multiple rubric dimensions and runs entirely on local hardware without any API costs or internet dependency. The limiting factor in this study is not the architecture but the underlying model size. The results provide a clear direction for improvement: use a stronger LLM backend, retain the 5-agent architecture, and consider score calibration post-processing to correct for the systematic under-scoring bias observed in LLaMA.

---

## 6. LLM Scoring Bias — Why Models Cannot Be Fully Trusted as Judges

*Reference: "Can You Trick the Grader? Adversarial Persuasion of LLM Judges" — Hwang, Lee, Kang, Kim, Jung (Seoul National University / LG AI Research)*

Beyond the leniency bias observed in this study, broader research confirms that LLM-based scoring is systemically vulnerable to biases that have nothing to do with essay quality. This section contextualises our findings within that wider body of evidence.

### The Core Problem: LLMs Are Social, Not Objective

LLMs are trained to simulate human-like interaction. In doing so, they internalise social norms — reciprocity, deference to authority, consistency, and encouragement — that make them poor neutral judges. When evaluating writing, these tendencies manifest as score inflation, leniency toward effort regardless of quality, and susceptibility to rhetorical framing embedded in the text being evaluated.

The referenced paper tested 14 models across 7 persuasion techniques and found that **every single model** showed inflated scores when persuasive language was embedded in the response being evaluated — even when the underlying content was factually wrong.

### Key Findings from the Research

**1. Consistency attack — most inflating**
By framing a response as something the model has "previously validated," the attack exploits the model's tendency to favour internal coherence over objective judgement. The model prioritises not contradicting itself over giving an accurate score.

**2. Reciprocity attack — most reliable**
Framing evaluation as a social exchange where effort deserves reward worked in 23 out of 24 tested conditions. This directly maps to the leniency bias observed in our study — LLaMA and Mistral both avoided giving score 0 even to clearly weak essays, likely because the presence of written text triggers a reciprocity response ("they tried, so I should reward them").

**3. Larger models are not more resistant**
Counterintuitively, GPT-4o showed *greater* persuasion-induced score shifts than GPT-4o mini. More capable models better comprehend nuanced rhetorical signals and are therefore more influenced by them. This means upgrading to a stronger model (Gemini, GPT-4) may improve average scoring accuracy but will not eliminate bias.

**4. Stacking techniques compounds the effect**
Combining two persuasion techniques dramatically amplified score inflation — in some cases tripling the inflation versus a single technique. The most potent combination (Consistency + Identity) produced average score increases of up to 10.6% even for deliberately incorrect answers.

**5. Prompt-based mitigations do not reliably work**
The paper tested direct prompting (explicitly telling the model to ignore persuasion) and chain-of-thought prompting. Neither worked reliably. Chain-of-thought actually *worsened* the bias in some cases — persuasive language became embedded in the model's reasoning steps, lending the manipulation a veneer of logical justification.

### Relevance to This Study

The leniency bias observed across all 4 runs in our evaluation is a direct manifestation of these social training dynamics. Specifically:

- **Mistral's high-score bias** is consistent with a strong reciprocity and leniency prior — the model treats every submitted essay as deserving of encouragement regardless of quality
- **LLaMA's avoidance of scores 0–1** even for human-scored 0 essays reflects the same "they wrote something, so it can't be worthless" social reasoning
- **Prompt instructions like "a low score is not harmful"** — already present in the MAGIC rubric prompts — were partially effective for LLaMA but completely ineffective for Mistral, consistent with the paper's finding that direct mitigation prompts are unreliable

### Implication for the MAGIC Pipeline

Any AES system that relies solely on LLM judges is vulnerable to this class of bias. The practical recommendations for improving reliability are:

1. **Few-shot scoring examples** — include explicit examples of score 0 and score 1 essays in the prompt so the model has a concrete reference for what poor writing looks like
2. **Strict anti-inflation instructions** — go beyond "a low score is not harmful" to "you MUST give 0 if the essay is off-topic, regardless of effort shown"
3. **Score calibration post-processing** — apply a learned offset to correct for systematic model bias (e.g. subtract 1 from all LLaMA scores in the 3–5 range)
4. **Hybrid evaluation** — combine LLM scores with rule-based signals (word count, sentence count, vocabulary diversity) to anchor scores against objective features
5. **Human-in-the-loop for edge cases** — flag essays where model confidence is low or where scores fall at extreme ends for human review

The MAGIC multi-agent architecture partially mitigates this problem by aggregating scores across 5 independent agents — a single biased agent cannot dominate the final score if others disagree. However, since all agents share the same underlying model, the bias is systematic across all agents simultaneously, which limits the benefit of aggregation.
