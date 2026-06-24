"""
llm_client.py — Unified LLM backend for Hybrid Pipeline

Merges Magic Pipeline's llm_client (call_llm, call_llm_orchestrator)
with AutoSCORE's llm_client (call_llm_raw, strip_markdown_fences).

The Hybrid needs both:
  - call_llm_raw()          → SRCE agent (returns raw string for JSON parsing)
  - call_llm()              → 5 MAGIC specialist agents (returns parsed dict)
  - call_llm_orchestrator() → Orchestrator (uses smarter model for Gemini)

Supports Gemini (cloud) and Ollama (local).
"""

import json
import re

# ── Module-level config ────────────────────────────────────────────────────────
_BACKEND:            str = "gemini"
_MODEL:              str = "gemini-2.0-flash"
_ORCHESTRATOR_MODEL: str = "gemini-2.0-flash"

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

def configure(
    backend: str,
    model: str,
    api_key: str | None = None,
    orchestrator_model: str | None = None,
) -> None:
    """
    Call once at startup before any LLM calls.

    Args:
        backend:            "gemini" or "ollama"
        model:              agent model name (used for SRCE + 5 MAGIC agents)
        api_key:            required for gemini, ignored for ollama
        orchestrator_model: optional separate model for orchestrator (Gemini only)
                            defaults to same as model if not provided
    """
    global _BACKEND, _MODEL, _ORCHESTRATOR_MODEL
    _BACKEND            = backend.lower()
    _MODEL              = model
    _ORCHESTRATOR_MODEL = orchestrator_model or model

    if _BACKEND == "gemini":
        try:
            import google.generativeai as genai
            if not api_key:
                raise ValueError("api_key is required for Gemini backend")
            genai.configure(api_key=api_key)
            print(
                f"  [LLM] ✅ Gemini configured  →  "
                f"agents: {_MODEL}  |  orchestrator: {_ORCHESTRATOR_MODEL}"
            )
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")

    elif _BACKEND == "ollama":
        _verify_ollama_connection(model)

    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'gemini' or 'ollama'.")


def call_llm_raw(system_prompt: str, user_prompt: str, label: str) -> str:
    """
    Send system + user prompt. Returns RAW string — caller handles JSON parsing.
    Used by the SRCE agent (returns a large JSON blob).
    """
    if _BACKEND == "gemini":
        return _call_gemini_raw(system_prompt, user_prompt, label)
    elif _BACKEND == "ollama":
        return _call_ollama_raw(system_prompt, user_prompt, label)
    else:
        return "{}"


def call_llm(prompt: str, label: str) -> dict:
    """
    Send a single combined prompt. Returns parsed {"score": int, "examiner_comment": str}.
    Used by the 5 MAGIC specialist agents.
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
    Send a prompt using the orchestrator model (may differ from agent model for Gemini).
    Returns parsed {"score": int, "examiner_comment": str}.
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


def strip_markdown_fences(raw: str) -> str:
    """
    Strips ```json ... ``` or ``` ... ``` wrappers if the model added them.
    Returns clean string ready for json.loads().
    """
    cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip()
    cleaned = cleaned.rstrip('`').strip()
    return cleaned


def _strip_thinking(raw: str) -> str:
    """
    Strips Qwen3-style <think>...</think> reasoning blocks from model output.
    These blocks appear before the actual JSON/score response and break parsing.
    This is a no-op for models that don't output thinking tokens (llama, mistral, etc.).
    """
    # Remove <think>...</think> blocks (including multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    return cleaned.strip()


# ── Gemini Backend ─────────────────────────────────────────────────────────────

def _call_gemini_raw(system_prompt: str, user_prompt: str, label: str) -> str:
    """
    Uses system_instruction= kwarg so SRCE gets true system-level context.
    Returns raw string — for large JSON evidence blobs.
    """
    try:
        import google.generativeai as genai
        model = genai.GenerativeModel(
            model_name=_MODEL,
            system_instruction=system_prompt,
        )
        response = model.generate_content(user_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"  [!] Gemini error for {label}: {e}")
        return "{}"


def _call_gemini(prompt: str, label: str, model: str | None = None) -> dict:
    """Calls Gemini with a combined prompt string. Returns parsed score dict."""
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
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()

        available  = [m["name"] for m in r.json().get("models", [])]
        model_base = model.split(":")[0]
        matched    = any(model_base in a for a in available)

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


def _call_ollama_raw(system_prompt: str, user_prompt: str, label: str) -> str:
    """
    For SRCE agent: concatenates system + user and requests a large output window.
    num_predict=2048 — evidence JSON can be 1k+ tokens.
    """
    import requests
    combined = (
        f"### System:\n{system_prompt}\n\n"
        f"### User:\n{user_prompt}\n\n"
        f"### Response:\n"
    )
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  _MODEL,
                "prompt": combined,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 4096,
                    "num_ctx": 32768,    # qwen tokenizer needs large context for SRCE prompts
                }
            },
            timeout=300,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        raw = _strip_thinking(raw)   # strip Qwen3 <think>...</think> blocks
        return raw
    except Exception as e:
        print(f"  [!] Ollama error for {label}: {e}")
        return "{}"


def _call_ollama(prompt: str, label: str) -> dict:
    """For MAGIC agents and orchestrator: uses shorter output window (512 tokens)."""
    import requests
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  _MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 512,
                    "num_ctx": 4096,
                }
            },
            timeout=180,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        raw = _strip_thinking(raw)   # strip Qwen3 <think>...</think> blocks
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
      3. Prose sentences: "I would score this essay a 5" — caught by fallback regex
    """
    # Primary: valid JSON object
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

    # Fallback 1: extract score and comment separately
    score_match = re.search(r'"score":\s*["\']?(\d)["\']?', raw)
    if score_match:
        comment_match = re.search(r'"examiner_comment"\s*:\s*"(.+)', raw, re.DOTALL)
        if comment_match:
            raw_comment = comment_match.group(1)
            clean = re.sub(r'"?\s*\}?\s*$', '', raw_comment).strip().rstrip('"')
            comment = clean if clean else raw
        else:
            comment = raw
        return {"score": int(score_match.group(1)), "examiner_comment": comment}

    # Fallback 2: prose patterns
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

    # Fallback 3: last resort
    last_resort = re.search(r'score[^.!\n]{0,30}([1-6])', raw, re.IGNORECASE)
    if last_resort:
        return {"score": int(last_resort.group(1)), "examiner_comment": raw}

    return {"score": 0, "examiner_comment": raw}
