"""
llm_client.py — Unified LLM backend for MAGIC pipeline

Supports:
  - Gemini  (google-generativeai, cloud)
  - Ollama  (local, 8b–27b models via REST API)

Usage:
    from llm_client import configure, call_llm

    configure(backend="gemini", model="gemini-2.0-flash", api_key="...")
    # or
    configure(backend="ollama", model="llama3.1:8b")

    result = call_llm(prompt, label="Agent 1")
    # result = {"score": int, "examiner_comment": str}
"""

import json
import re

# ── Module-level config (set once at startup via configure()) ──────────────────
_BACKEND:             str = "gemini"
_MODEL:               str = "gemini-2.5-flash"
_ORCHESTRATOR_MODEL:  str = "gemini-2.5-pro"   # smarter model for orchestrator synthesis

GEMINI_DEFAULTS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

OLLAMA_DEFAULTS = [
    "llama3.1:8b",
    "llama3.2:3b",
    "gemma3:12b",
    "gemma3:27b",
    "mistral:7b",
    "phi4:14b",
]


# ── Public API ─────────────────────────────────────────────────────────────────

def configure(backend: str, model: str, api_key: str | None = None,
              orchestrator_model: str | None = None) -> None:
    """
    Call this once at startup before any call_llm() calls.

    Args:
        backend:            "gemini" or "ollama"
        model:              agent model name (used for all 5 specialist agents)
        api_key:            required for gemini, ignored for ollama
        orchestrator_model: optional separate model for orchestrator synthesis
                            defaults to same as model if not provided
    """
    global _BACKEND, _MODEL, _ORCHESTRATOR_MODEL
    _BACKEND = backend.lower()
    _MODEL   = model
    _ORCHESTRATOR_MODEL = orchestrator_model or model

    if _BACKEND == "gemini":
        try:
            import google.generativeai as genai
            if not api_key:
                raise ValueError("api_key is required for Gemini backend")
            genai.configure(api_key=api_key)
            print(f"  [LLM] ✅ Gemini configured  →  agents: {_MODEL}  |  orchestrator: {_ORCHESTRATOR_MODEL}")
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")

    elif _BACKEND == "ollama":
        _verify_ollama_connection(model)

    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'gemini' or 'ollama'.")


def call_llm(prompt: str, label: str) -> dict:
    """
    Send a prompt using the agent model (gemini-2.5-flash).
    Returns {"score": int, "examiner_comment": str}.
    Never raises — returns score=0 on any error.
    """
    if _BACKEND == "gemini":
        return _call_gemini(prompt, label, model=_MODEL)
    elif _BACKEND == "ollama":
        return _call_ollama(prompt, label)
    else:
        return {"score": 0, "examiner_comment": f"Error: unconfigured backend '{_BACKEND}'"}


def call_llm_orchestrator(prompt: str, label: str = "Orchestrator") -> dict:
    """
    Send a prompt using the orchestrator model (gemini-2.5-pro).
    Falls back to agent model for ollama (no separate orchestrator model there).
    Returns {"score": int, "examiner_comment": str}.
    Never raises — returns score=0 on any error.
    """
    if _BACKEND == "gemini":
        return _call_gemini(prompt, label, model=_ORCHESTRATOR_MODEL)
    elif _BACKEND == "ollama":
        return _call_ollama(prompt, label)
    else:
        return {"score": 0, "examiner_comment": f"Error: unconfigured backend '{_BACKEND}'"}


def get_backend_info() -> dict:
    """Returns current backend/model config for display."""
    return {"backend": _BACKEND, "model": _MODEL}


# ── Gemini Backend ─────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, label: str, model: str | None = None) -> dict:
    try:
        import google.generativeai as genai
        m        = genai.GenerativeModel(model or _MODEL)
        response = m.generate_content(prompt)
        raw      = response.text.strip()
        return _parse_response(raw, label)
    except Exception as e:
        print(f"  [!] Gemini error for {label}: {e}")
        return {"score": 0, "examiner_comment": f"Error: {str(e)}"}


# ── Ollama Backend ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"


def _verify_ollama_connection(model: str) -> None:
    """Checks Ollama is running and the requested model is available."""
    import requests
    try:
        # Check server is up
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()

        # Check model is pulled
        available = [m["name"] for m in r.json().get("models", [])]
        # Normalize: "llama3.1:8b" matches "llama3.1:8b" in list
        model_base = model.split(":")[0]
        matched = any(model_base in a for a in available)

        if not matched:
            print(f"\n  [!] Model '{model}' not found locally.")
            print(f"      Available: {available}")
            print(f"      Pull it with:  ollama pull {model}\n")
        else:
            print(f"  [LLM] ✅ Ollama configured  →  model: {model}  (server: {OLLAMA_BASE_URL})")

    except requests.exceptions.ConnectionError:
        print(f"\n  [!] Ollama server not running at {OLLAMA_BASE_URL}")
        print("      Start it with:  ollama serve\n")
    except Exception as e:
        print(f"  [!] Ollama check failed: {e}")


def _call_ollama(prompt: str, label: str) -> dict:
    """
    Calls Ollama's /api/generate endpoint.
    Uses non-streaming mode for simplicity.
    """
    import requests
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  _MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,   # low temp for consistent scoring
                    "num_predict": 512,   # enough for score + comment
                }
            },
            timeout=180,   # 3 min — larger models may need more time
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        return _parse_response(raw, label)

    except Exception as e:
        print(f"  [!] Ollama error for {label}: {e}")
        return {"score": 0, "examiner_comment": f"Error: {str(e)}"}


# ── Shared Response Parser ─────────────────────────────────────────────────────

def _parse_response(raw: str, label: str) -> dict:
    """
    Extracts {"score": int, "examiner_comment": str} from a raw LLM string.

    Handles 3 formats:
      1. Clean JSON:      {"score": 5, "examiner_comment": "..."}
      2. Keyed prose:     "score": 5  (without full valid JSON)
      3. Prose sentences: Llama often writes step-by-step markdown like
                          "I would score this essay a 5" — caught by fallback regex.
    """
    # ── Primary: valid JSON object ─────────────────────────────────────────────
    # Use GREEDY .*  (not .*?) so we capture the full JSON even with long comments
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return {
                "score":            int(result.get("score", 0)),
                "examiner_comment": str(result.get("examiner_comment", "No comment provided")),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Fallback 1: extract score and comment separately ───────────────────────
    score_match = re.search(r'"score":\s*["\']?(\d)["\']?', raw)
    if score_match:
        # Grab everything after "examiner_comment": " up to the end of the string,
        # then strip the trailing }" if present — handles embedded quotes in text
        comment_match = re.search(r'"examiner_comment"\s*:\s*"(.+)', raw, re.DOTALL)
        if comment_match:
            raw_comment = comment_match.group(1)
            # Strip trailing JSON closing characters: optional whitespace, ", }
            clean = re.sub(r'"?\s*\}?\s*$', '', raw_comment).strip().rstrip('"')
            comment = clean if clean else raw
        else:
            comment = raw
        return {"score": int(score_match.group(1)), "examiner_comment": comment}

    # ── Fallback 2: prose patterns (Llama step-by-step markdown) ──────────────
    prose_patterns = [
        r'score[d]?\s+(?:this\s+essay\s+)?(?:a\s+|of\s+)?([1-6])',
        r'assign\s+(?:a\s+)?score\s+of\s+([1-6])',
        r'give\s+(?:this\s+)?(?:essay\s+)?(?:a\s+)?(?:score\s+of\s+)?([1-6])',
        r'rated?\s+(?:a\s+)?([1-6])\s*(?:/\s*6|out)',
        r'([1-6])\s+out\s+of\s+6',
    ]
    for pattern in prose_patterns:
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            return {"score": int(m.group(1)), "examiner_comment": raw}

    # ── Fallback 3: last resort ────────────────────────────────────────────────
    last_resort = re.search(r'score[^.!\n]{0,30}([1-6])', raw, re.IGNORECASE)
    if last_resort:
        return {"score": int(last_resort.group(1)), "examiner_comment": raw}

    return {"score": 0, "examiner_comment": raw}
