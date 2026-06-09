# MAGIC Prototype — Quick Start Guide

---

## Step 1 · Get Your Free Gemini API Key

1. Go to **[https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)**
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (starts with `AIza...`)

> **Free tier limits:** 15 requests/min, 1,500 requests/day — more than enough for this prototype.

---

## Step 2 · Add the Key to `.env`

Open `n:\Dev\Summer Intern\Magic Prototype\.env` and replace the placeholder:

```env
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Save the file. That's the only config you need.

---

## Step 3 · Run the Pipeline

Open a terminal in the project folder and run:

```bash
cd "n:\Dev\Summer Intern\Magic Prototype"
python magic_graph.py
```

The pipeline will:
1. Fan out to **5 parallel agents** (you'll see Gemini being called)
2. Fan in to the **orchestrator**
3. Print the full **Rich terminal UI** with all scores + feedback

---

## Step 4 · Try the 3 Example Essays

Open `magic_graph.py` and replace the `ESSAY_TEXT` variable with one of the essays below.
The `ESSAY_PROMPT` stays the same for all three.

---

### Example A — Strong Essay (Expected Score: 5–6)

```python
ESSAY_TEXT = """
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
"""
```

---

### Example B — Average Essay (Expected Score: 3–4)

```python
ESSAY_TEXT = """
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
"""
```

---

### Example C — Weak Essay (Expected Score: 1–2)

```python
ESSAY_TEXT = """
I think the city should get bikes because bikes are good. Other cities did it and traffic went
down so we should do it too. It makes sense because if people ride bikes they won't drive cars.

There might be some problems but overall it is a good idea. The city should just try it and
see what happens. Bikes are also good for the environment and health so there are many reasons
to do it.
"""
```

---

## Switching Between Essays

In `magic_graph.py`, locate this block near the top:

```python
ESSAY_TEXT = """
...your essay here...
"""
```

Paste in whichever example you want, save the file, then run:

```bash
python magic_graph.py
```

---

## What the Output Looks Like

```
═══════════════ ✨ MAGIC Essay Evaluation Pipeline ✨ ════════════════

◆  Step 1–5 · Specialised Agent Scores

╭────┬──────────────┬──────────────────────────────────────────────┬────────────────────╮
│ #  │ Agent        │ Aspect                                       │ Score              │
├────┼──────────────┼──────────────────────────────────────────────┼────────────────────┤
│ 1  │ 📝 Agent 1   │ Aspect 1: Quality of the response...         │ ██████████ 5/6     │
│ 2  │ 🧠 Agent 2   │ Aspect 2: Considering the complexities...    │ ████████░░ 4/6     │
│ 3  │ 📐 Agent 3   │ Aspect 3: Organizing, developing...          │ ████████░░ 4/6     │
│ 4  │ 📚 Agent 4   │ Aspect 4: Vocabulary and sentence variety    │ ██████████ 5/6     │
│ 5  │ ✏️  Agent 5   │ Aspect 5: Grammar and mechanics              │ ██████████ 5/6     │
╰────┴──────────────┴──────────────────────────────────────────────┴────────────────────╯

◆  Detailed Reasoning per Agent
[coloured panels with step-by-step reasoning per agent]

━━━━━━━━━━━━━━━━━━━━━━ Score Summary ━━━━━━━━━━━━━━━━━━━━━━━

  📝 T1 · Prompt Response     ██████████ 5/6
  🧠 T2 · Argumentation       ████████░░ 4/6
  📐 T3 · Organisation        ████████░░ 4/6
  📚 T4 · Vocabulary          ██████████ 5/6
  ✏️  T5 · Grammar             ██████████ 5/6

━━━━━━━━━━━━━━━━ 🎯  Step 6 · Orchestrator Final Score ━━━━━━━━━━━━━

╭─────────────────────── 🎯  Final Evaluation ────────────────────────╮
│                                                                      │
│   FINAL HOLISTIC SCORE:  5 / 6                                      │
│   ██████████ 5/6                                                    │
│                                                                      │
│   Orchestrator Feedback:                                             │
│   [Combined, student-facing feedback from all 5 agents]             │
│                                                                      │
╰─────────────────────────────────────────────────────────────────────╯

━━━━━━━━━━━━━━━━━━━━━ End of MAGIC Evaluation ━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `KeyError: GOOGLE_API_KEY` | Check `.env` exists and key is correct |
| `ModuleNotFoundError: prompts` | Run from the `Magic Prototype` folder, not a subfolder |
| Score always 0 | Add `print(raw)` inside `call_llm()` to see raw model output |
| `JSONDecodeError` | Already handled by regex fallback — check raw output if score is 0 |
| Slow execution | Normal — 6 total API calls. Agents 1–5 run in parallel so total time ≈ one agent call + orchestrator |

---

## Project is 100% Complete ✅

| Component | Status |
|---|---|
| `prompts/__init__.py` | ✅ Done |
| `prompts/agents.py` | ✅ Import fixed |
| `prompts/orchestrator.py` | ✅ Import fixed |
| `magic_graph.py` | ✅ LangGraph pipeline with parallel agents |
| `display.py` | ✅ Rich terminal UI |
| `.env` | ✅ Ready for your key |
| `docs/implementation.md` | ✅ Full architecture doc |
| `docs/quickstart.md` | ✅ This file |
