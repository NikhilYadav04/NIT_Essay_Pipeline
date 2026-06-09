"""
batch_runner.py — Automated Batch Evaluation: MAGIC vs AutoSCORE
================================================================

Default backend: Ollama (llama3.1:8b) — local, no rate limits.

Iterates over every essay in data/asap_clean.csv, scores each essay
using BOTH pipelines (MAGIC and AutoSCORE), and writes:
  - results/batch_results.csv   ← structured scores (one row per essay)
  - results/trial_log.jsonl     ← full per-trial log (scores, agent
                                   breakdowns, reasoning, errors, timing)
  - results/run_log.txt         ← lightweight run-level event log

Features:
  - Default: Ollama / llama3.1:8b  (no API key or rate-limit needed)
  - Auto-resume: skips essays already scored if results CSV exists
  - Per-essay error isolation — one failure doesn't stop the run
  - Full per-trial JSONL log for traceability and debugging

Usage:
    # Default: Ollama + llama3.1:8b
    python batch_eval/batch_runner.py

    # Switch to Gemini
    python batch_eval/batch_runner.py --backend gemini --model gemini-2.0-flash

    # Quick smoke test on first 3 essays
    python batch_eval/batch_runner.py --limit 3

    # MAGIC only
    python batch_eval/batch_runner.py --pipeline magic

    # Resume an interrupted run
    python batch_eval/batch_runner.py --resume
"""

import os
import sys
import time
import json
import argparse
import traceback
import pandas as pd
from datetime import datetime

import importlib.util

# ── Path setup ────────────────────────────────────────────────────────────────
BATCH_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(BATCH_DIR)  # n:\Dev\Summer Intern

MAGIC_DIR        = os.path.join(ROOT, "Magic Prototype")
AUTOSCORE_DIR    = os.path.join(ROOT, "AutoScore Pipeline")
AUTOSCORE_SHARED = os.path.join(AUTOSCORE_DIR, "shared")

# NOTE: We do NOT add MAGIC_DIR or AUTOSCORE_DIR to sys.path globally because
# both contain a 'prompts/' sub-package and they would shadow each other.
# Instead we load each graph with importlib + a temporary cwd swap so their
# relative imports resolve correctly.
sys.path.insert(0, AUTOSCORE_SHARED)   # shared llm_client only

# ── File paths ─────────────────────────────────────────────────────────────────
DATA_FILE    = os.path.join(BATCH_DIR, "data", "asap_clean.csv")
RESULTS_FILE = os.path.join(BATCH_DIR, "results", "batch_results.csv")
TRIAL_LOG    = os.path.join(BATCH_DIR, "results", "trial_log.jsonl")   # full per-essay log
LOG_FILE     = os.path.join(BATCH_DIR, "results", "run_log.txt")        # lightweight event log

# ── ASAP rubric for AutoSCORE ──────────────────────────────────────────────────
ASAP_RUBRIC = """
Trait 1: Task Response
  Score 6: Clear, insightful position that fully addresses the prompt with strong elaboration
  Score 5: Clear, well-considered position addressing the prompt
  Score 4: Clear position addressing the prompt with adequate development
  Score 3: Vague or limited in addressing the prompt
  Score 2: Unclear or seriously limited response to the prompt
  Score 1: Little or no understanding of how to respond
  Score 0: Off-topic, blank, or incomprehensible

Trait 2: Argument Quality
  Score 6: Fully developed with compelling reasons and persuasive evidence
  Score 5: Developed with logically sound reasons and well-chosen examples
  Score 4: Developed with relevant reasons and/or examples
  Score 3: Weak use of reasons/examples, relies on unsupported claims
  Score 2: Few or no relevant reasons or examples
  Score 1: Little or no evidence of understanding the issue
  Score 0: Off-topic, blank, or incomprehensible

Trait 3: Organisation
  Score 6: Well-focused, well-organised, connecting ideas logically throughout
  Score 5: Focused and generally well-organised
  Score 4: Adequately focused and organised
  Score 3: Limited in focus and/or organisation
  Score 2: Poorly focused and/or poorly organised
  Score 1: Little or no organised response
  Score 0: Off-topic, blank, or incomprehensible

Trait 4: Language & Style
  Score 6: Fluent and precise, effective vocabulary and sentence variety
  Score 5: Clear and well-expressed, appropriate vocabulary and variety
  Score 4: Acceptable clarity, sufficient language control
  Score 3: Language problems causing lack of clarity
  Score 2: Serious language problems frequently interfering with meaning
  Score 1: Severe language problems persistently interfering with meaning
  Score 0: Off-topic, blank, or incomprehensible

Trait 5: Grammar & Mechanics
  Score 6: Superior facility with conventions — minor errors only
  Score 5: Good facility with conventions — occasional minor errors
  Score 4: General control of conventions — some errors
  Score 3: Occasional major or frequent minor errors interfering with meaning
  Score 2: Serious errors frequently obscuring meaning
  Score 1: Pervasive errors resulting in incoherence
  Score 0: Off-topic, blank, or incomprehensible
""".strip()


# ── Column definitions for results CSV ────────────────────────────────────────
RESULT_COLS = [
    "essay_id", "dataset", "human_score_raw", "human_score_norm",
    "magic_score", "magic_error",
    "autoscore_holistic", "autoscore_trait1", "autoscore_trait2",
    "autoscore_trait3", "autoscore_trait4", "autoscore_trait5",
    "autoscore_error",
    "magic_duration_sec", "autoscore_duration_sec",
    "timestamp",
]


def log(msg: str, also_print: bool = True):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_trial(trial: dict):
    """Append one complete trial record as a JSON line to trial_log.jsonl."""
    with open(TRIAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(trial, ensure_ascii=False) + "\n")


# ── MAGIC invocation ──────────────────────────────────────────────────────────

def run_magic(magic_app, essay_text: str, essay_prompt: str) -> tuple:
    """
    Returns (orchestrator_score, agent_results_list, error_message, duration_sec).
    orchestrator_score = -1 on error, agent_results_list = [] on error.
    """
    try:
        t0 = time.time()
        final_state = magic_app.invoke({
            "essay_text":    essay_text,
            "essay_prompt":  essay_prompt,
            "agent_results": [],
        })
        duration = round(time.time() - t0, 1)
        score  = final_state.get("orchestrator_score", -1)
        agents = sorted(
            final_state.get("agent_results", []),
            key=lambda r: r.get("agent_index", 0)
        )
        return score, agents, "", duration
    except Exception as e:
        return -1, [], str(e)[:300], 0.0


# ── AutoSCORE invocation ───────────────────────────────────────────────────────

def run_autoscore(autoscore_app, essay_text: str, essay_prompt: str) -> tuple:
    """
    Returns (evidence_dict, scoring_dict, error_message, duration).
    Both dicts are empty on error.
    """
    try:
        t0 = time.time()
        final_state = autoscore_app.invoke({
            "essay_text":    essay_text,
            "essay_prompt":  essay_prompt,
            "rubric":        ASAP_RUBRIC,
            "evidence_dict": {},
            "evidence_json": "",
            "scoring_dict":  {},
        })
        duration = round(time.time() - t0, 1)
        return (
            final_state.get("evidence_dict", {}),
            final_state.get("scoring_dict",  {}),
            "",
            duration,
        )
    except Exception as e:
        return {}, {}, str(e)[:300], 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_autoscore_traits(scoring_dict: dict) -> dict:
    """Pull per-trait scores from AutoSCORE's scoring_dict safely."""
    trait_keys = {
        "autoscore_trait1": ["trait1", "trait_1", "task_response", "Trait 1"],
        "autoscore_trait2": ["trait2", "trait_2", "argument_quality", "Trait 2"],
        "autoscore_trait3": ["trait3", "trait_3", "organisation", "Trait 3"],
        "autoscore_trait4": ["trait4", "trait_4", "language_style", "Trait 4"],
        "autoscore_trait5": ["trait5", "trait_5", "grammar_mechanics", "Trait 5"],
    }

    # Flatten nested score values
    def get_score(d, keys):
        for k in keys:
            if k in d:
                v = d[k]
                if isinstance(v, dict):
                    return v.get("score", v.get("Score", -1))
                if isinstance(v, (int, float)):
                    return int(v)
        # Try lowercase search
        for k in d:
            for alias in keys:
                if alias.lower() in k.lower():
                    v = d[k]
                    if isinstance(v, dict):
                        return v.get("score", v.get("Score", -1))
                    if isinstance(v, (int, float)):
                        return int(v)
        return -1

    result = {}
    for col, keys in trait_keys.items():
        result[col] = get_score(scoring_dict, keys)

    # holistic_score
    holistic = -1
    for k in ["holistic_score", "holistic", "overall_score", "final_score", "total_score"]:
        if k in scoring_dict:
            v = scoring_dict[k]
            holistic = int(v) if isinstance(v, (int, float)) else -1
            break

    result["autoscore_holistic"] = holistic
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Evaluation: MAGIC vs AutoSCORE on ASAP-100")
    parser.add_argument("--backend",  choices=["gemini", "ollama"], default="ollama",
                        help="LLM backend (default: ollama)")
    parser.add_argument("--model",    type=str, default=None,
                        help="Model override (default: ollama→llama3.1:8b  gemini→gemini-2.5-flash)")
    parser.add_argument("--orchestrator-model", type=str, default=None, dest="orchestrator_model",
                        help="Separate model for orchestrator (default: gemini-2.5-pro for gemini backend)")
    parser.add_argument("--pipeline", choices=["both", "magic", "autoscore"], default="both",
                        help="Which pipeline(s) to run (default: both)")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Evaluate only the first N essays (for quick testing)")
    parser.add_argument("--rubric",   choices=["gre", "asap"], default="gre",
                        help="MAGIC rubric variant: gre (5 agents) or asap (3 agents, default: gre)")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip essays that are already in batch_results.csv")
    parser.add_argument("--sleep",    type=float, default=1.0,
                        help="Seconds to sleep between pipeline calls (default: 1 for Ollama)")
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Resolve model ──────────────────────────────────────────────────────────
    default_models = {"gemini": "gemini-2.5-flash", "ollama": "llama3.1:8b"}
    model = args.model or default_models[args.backend]
    orchestrator_model = args.orchestrator_model or ("gemini-2.5-pro" if args.backend == "gemini" else None)

    print("\n" + "=" * 65)
    print("  MAGIC vs AutoSCORE — Batch Evaluation Pipeline")
    print("=" * 65)
    print(f"  Backend  : {args.backend.upper()}")
    print(f"  Model    : {model}  (orchestrator: {orchestrator_model or model})")
    print(f"  Pipeline : {args.pipeline}")
    print(f"  Rubric   : {args.rubric} ({'3 agents' if args.rubric == 'asap' else '5 agents'})")
    print(f"  Sleep    : {args.sleep}s between essays")
    print(f"  Resume   : {args.resume}")
    print("=" * 65 + "\n")

    # ── Ensure dirs exist ──────────────────────────────────────────────────────
    os.makedirs(os.path.join(BATCH_DIR, "data"),    exist_ok=True)
    os.makedirs(os.path.join(BATCH_DIR, "results"), exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────────────
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Clean dataset not found: {DATA_FILE}")
        print("  Run prepare_data.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    if args.limit:
        df = df.head(args.limit)

    total = len(df)
    print(f"[1] Loaded {total} essays from {os.path.basename(DATA_FILE)}\n")

    # ── Resume logic ───────────────────────────────────────────────────────────
    completed_ids = set()
    existing_rows = []

    if args.resume and os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE)
        completed_ids = set(existing["essay_id"].tolist())
        existing_rows = existing.to_dict("records")
        print(f"[2] Resuming — {len(completed_ids)} essays already scored, skipping them.\n")
    else:
        # Fresh run — clear previous logs so they don't mix
        for f in [TRIAL_LOG, LOG_FILE]:
            if os.path.exists(f):
                open(f, "w").close()
        print(f"[2] Starting fresh run.\n")

    # ── Configure LLM ─────────────────────────────────────────────────────────
    api_key = os.environ.get("GOOGLE_API_KEY") if args.backend == "gemini" else None

    # ── Build pipeline graphs ──────────────────────────────────────────────────
    # Both pipelines have a 'prompts/' sub-package — adding both roots to
    # sys.path causes a collision. We load each graph via importlib with a
    # temporary cwd+path swap so their internal imports resolve in isolation.
    # After each graph loads, we configure the llm_client it pulled into
    # sys.modules so all internal calls use the right backend.
    magic_app      = None
    autoscore_app  = None
    _orig_cwd = os.getcwd()

    def _evict_prompts():
        """Remove all cached 'prompts' and 'prompts.*' entries from sys.modules
        so the next pipeline loads its own fresh copy."""
        to_remove = [k for k in sys.modules if k == "prompts" or k.startswith("prompts.")]
        for k in to_remove:
            del sys.modules[k]

    def _load_module_from_dir(module_name: str, file_path: str, project_dir: str):
        """Load a module from a specific file path with project_dir on sys.path.
        Evicts any cached 'prompts' package first so each pipeline gets its own."""
        _evict_prompts()
        # Also remove the graph module itself in case it was registered before
        sys.modules.pop(module_name, None)

        sys.path.insert(0, project_dir)
        os.chdir(project_dir)
        try:
            spec   = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        finally:
            while project_dir in sys.path:
                sys.path.remove(project_dir)
            os.chdir(_orig_cwd)

    if args.pipeline in ("both", "magic"):
        magic_module = _load_module_from_dir(
            "magic_graph",
            os.path.join(MAGIC_DIR, "magic_graph.py"),
            MAGIC_DIR,
        )
        llm_mod = sys.modules.get("llm_client")
        if llm_mod:
            llm_mod.configure(backend=args.backend, model=model, api_key=api_key,
                              orchestrator_model=orchestrator_model)
        magic_app = magic_module.build_graph(rubric=args.rubric)
        print(f"[3a] MAGIC graph compiled ✓ (rubric={args.rubric})")

    if args.pipeline in ("both", "autoscore"):
        # Evict MAGIC's prompts before loading AutoSCORE's
        autoscore_module = _load_module_from_dir(
            "autoscore_graph",
            os.path.join(AUTOSCORE_DIR, "autoscore_graph.py"),
            AUTOSCORE_DIR,
        )
        llm_mod = sys.modules.get("llm_client")
        if llm_mod:
            llm_mod.configure(backend=args.backend, model=model, api_key=api_key)
        autoscore_app = autoscore_module.build_graph()
        print("[3b] AutoSCORE graph compiled ✓")

    print()

    # ── Evaluation loop ────────────────────────────────────────────────────────
    results = list(existing_rows)  # carry over resumed rows

    log(f"Run started — backend={args.backend} model={model} essays={total}")

    for idx, row in df.iterrows():
        essay_id   = str(row["essay_id"])     # ASAP IDs are alphanumeric strings
        dataset    = row["dataset"]
        essay_text = str(row["essay_text"])
        prompt     = str(row["essay_prompt"])
        h_raw      = int(row["human_score_raw"])
        h_norm     = int(row["human_score_norm"])
        n          = df.index.get_loc(idx) + 1  # 1-based display index

        # Skip if resumed
        if essay_id in completed_ids:
            continue

        print(f"  [{n:>3}/{total}] Essay {essay_id:>5} | Human: {h_norm}/6 ({h_raw} raw)", end="  ")

        result_row = {
            "essay_id":             essay_id,
            "dataset":              dataset,
            "human_score_raw":      h_raw,
            "human_score_norm":     h_norm,
            "magic_score":          -1,
            "magic_error":          "",
            "autoscore_holistic":   -1,
            "autoscore_trait1":     -1,
            "autoscore_trait2":     -1,
            "autoscore_trait3":     -1,
            "autoscore_trait4":     -1,
            "autoscore_trait5":     -1,
            "autoscore_error":      "",
            "magic_duration_sec":   0.0,
            "autoscore_duration_sec": 0.0,
            "timestamp":            datetime.now().isoformat(timespec="seconds"),
        }

        # ── MAGIC ──────────────────────────────────────────────────────────────
        magic_agents   = []   # full per-agent breakdown for trial log
        magic_feedback = ""

        if magic_app:
            score, magic_agents, err, dur = run_magic(magic_app, essay_text, prompt)
            result_row["magic_score"]        = score
            result_row["magic_error"]        = err
            result_row["magic_duration_sec"] = dur

            if err:
                log(f"  Essay {essay_id}: MAGIC error — {err}", also_print=False)
                print(f"MAGIC=ERR  ", end="")
            else:
                print(f"MAGIC={score}/6  ", end="")

            time.sleep(args.sleep)

        # ── AutoSCORE ──────────────────────────────────────────────────────────
        evidence_dict = {}
        scoring_dict  = {}

        if autoscore_app:
            evidence_dict, scoring_dict, err, dur = run_autoscore(autoscore_app, essay_text, prompt)
            traits = extract_autoscore_traits(scoring_dict)

            result_row.update(traits)
            result_row["autoscore_error"]          = err
            result_row["autoscore_duration_sec"]   = dur

            if err:
                log(f"  Essay {essay_id}: AutoSCORE error — {err}", also_print=False)
                print(f"AutoSCORE=ERR")
            else:
                print(f"AutoSCORE={result_row['autoscore_holistic']}/6")

            time.sleep(args.sleep)
        else:
            print()

        results.append(result_row)

        # ── Save results CSV (checkpointed after every essay) ──────────────────
        pd.DataFrame(results)[RESULT_COLS].to_csv(RESULTS_FILE, index=False)

        # ── Write detailed per-trial log entry ─────────────────────────────────
        trial_record = {
            # ── Identity ──────────────────────────────────────────────────────
            "timestamp":        result_row["timestamp"],
            "essay_id":         essay_id,
            "dataset":          dataset,
            "trial_index":      n,
            "total_essays":     total,

            # ── Human score ───────────────────────────────────────────────────
            "human_score_raw":  h_raw,
            "human_score_norm": h_norm,

            # ── Essay metadata ────────────────────────────────────────────────
            "word_count":       int(row.get("word_count", 0)),
            "essay_snippet":    essay_text[:200].replace("\n", " "),  # first 200 chars

            # ── MAGIC results ─────────────────────────────────────────────────
            "magic": {
                "orchestrator_score": result_row["magic_score"],
                "duration_sec":       result_row["magic_duration_sec"],
                "error":              result_row["magic_error"],
                "agents": [
                    {
                        "index":   a.get("agent_index"),
                        "aspect":  a.get("aspect_name"),
                        "type":    a.get("agent_type"),
                        "score":   a.get("score"),
                        "comment": a.get("examiner_comment", "")[:300],  # truncate long comments
                    }
                    for a in magic_agents
                ],
            },

            # ── AutoSCORE results ─────────────────────────────────────────────
            "autoscore": {
                "holistic_score":  result_row["autoscore_holistic"],
                "trait_scores": {
                    "trait1_task_response":    result_row["autoscore_trait1"],
                    "trait2_argument_quality": result_row["autoscore_trait2"],
                    "trait3_organisation":     result_row["autoscore_trait3"],
                    "trait4_language_style":   result_row["autoscore_trait4"],
                    "trait5_grammar":          result_row["autoscore_trait5"],
                },
                "evidence_summary": {
                    # Snapshot key fields from SRCE agent's evidence dict
                    k: (str(v)[:200] if isinstance(v, str) else v)
                    for k, v in evidence_dict.items()
                    if k in [
                        "task_response", "argument_quality", "organisation",
                        "language_style", "grammar_mechanics",
                        "trait1", "trait2", "trait3", "trait4", "trait5",
                    ]
                },
                "duration_sec": result_row["autoscore_duration_sec"],
                "error":        result_row["autoscore_error"],
            },

            # ── Comparison delta ──────────────────────────────────────────────
            "delta": {
                "magic_vs_human":     result_row["magic_score"] - h_norm
                                      if result_row["magic_score"] >= 0 else None,
                "autoscore_vs_human": result_row["autoscore_holistic"] - h_norm
                                      if result_row["autoscore_holistic"] >= 0 else None,
                "magic_vs_autoscore": result_row["magic_score"] - result_row["autoscore_holistic"]
                                      if (result_row["magic_score"] >= 0
                                          and result_row["autoscore_holistic"] >= 0) else None,
            },
        }

        log_trial(trial_record)

    # ── Final summary ──────────────────────────────────────────────────────────
    final_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("  RUN COMPLETE")
    print("=" * 65)
    print(f"  Total essays scored : {len(final_df)}")

    valid_magic = final_df[final_df["magic_score"] >= 0]
    valid_auto  = final_df[final_df["autoscore_holistic"] >= 0]

    if len(valid_magic) > 0:
        magic_errors = len(final_df) - len(valid_magic)
        print(f"  MAGIC       : {len(valid_magic)} ok, {magic_errors} errors | Mean score: {valid_magic['magic_score'].mean():.2f}")

    if len(valid_auto) > 0:
        auto_errors = len(final_df) - len(valid_auto)
        print(f"  AutoSCORE   : {len(valid_auto)} ok, {auto_errors} errors | Mean score: {valid_auto['autoscore_holistic'].mean():.2f}")

    print(f"\n  Results saved → {RESULTS_FILE}")
    print("  Next step: run analyze.py to compute QWK, Pearson, MAE\n")

    log(f"Run finished — {len(final_df)} essays scored")


if __name__ == "__main__":
    # Ensure .env is loaded from Magic Prototype
    from dotenv import load_dotenv
    load_dotenv(os.path.join(MAGIC_DIR, ".env"))
    main()
