"""
llm_client.py — Unified LLM backend for AutoSCORE pipeline

Supports:
  - Gemini  (google-generativeai, cloud)
  - Ollama  (local, 8b–27b models via REST API)

Key difference from MAGIC's llm_client:
  call_llm_raw() returns a raw string instead of a parsed dict —
  AutoSCORE agents return large JSON blobs that need custom parsing.

Usage:
    from shared.llm_client import configure, call_llm_raw

    configure(backend="gemini", model="gemini-2.0-flash", api_key="...")
    # or
    configure(backend="ollama", model="llama3.1:8b")

    raw = call_llm_raw(system_prompt, user_prompt, label="SRCE Agent")
"""

import re

# ── Module-level config (set once at startup via configure()) ──────────────────
_BACKEND: str = "gemini"
_MODEL:   str = "gemini-2.0-flash"

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

def configure(backend: str, model: str, api_key: str | None = None) -> None:
    """
    Call once at startup before any call_llm_raw() calls.

    Args:
        backend:  "gemini" or "ollama"
        model:    model name string
        api_key:  required for gemini, ignored for ollama
    """
    global _BACKEND, _MODEL
    _BACKEND = backend.lower()
    _MODEL   = model

    if _BACKEND == "gemini":
        try:
            import google.generativeai as genai
            if not api_key:
                raise ValueError("api_key is required for Gemini backend")
            genai.configure(api_key=api_key)
            print(f"  [LLM] ✅ Gemini configured  →  model: {_MODEL}")
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")

    elif _BACKEND == "ollama":
        _verify_ollama_connection(model)

    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'gemini' or 'ollama'.")


def call_llm_raw(system_prompt: str, user_prompt: str, label: str) -> str:
    """
    Send a system + user prompt to the configured backend.
    Returns the raw response string — caller handles JSON parsing.

    AutoSCORE agents return large structured JSON blobs,
    so raw string is more appropriate than a parsed dict.
    """
    if _BACKEND == "gemini":
        return _call_gemini_raw(system_prompt, user_prompt, label)
    elif _BACKEND == "ollama":
        return _call_ollama_raw(system_prompt, user_prompt, label)
    else:
        print(f"  [!] Unconfigured backend '{_BACKEND}'")
        return "{}"


def get_backend_info() -> dict:
    """Returns current backend/model config for display."""
    return {"backend": _BACKEND, "model": _MODEL}


# ── Gemini Backend ─────────────────────────────────────────────────────────────

def _call_gemini_raw(system_prompt: str, user_prompt: str, label: str) -> str:
    """
    Uses system_instruction= kwarg so Agent 1 and Agent 2 get true
    system-level context, not just a prepended string.
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
    Ollama /api/generate is single-turn, so we concatenate system + user.
    num_predict is set high (2048) since SRCE + Scoring JSON blobs are large.
    """
    import requests
    combined_prompt = (
        f"### System:\n{system_prompt}\n\n"
        f"### User:\n{user_prompt}\n\n"
        f"### Response:\n"
    )
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  _MODEL,
                "prompt": combined_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,    # very low — we want deterministic JSON
                    "num_predict": 2048,   # large — evidence JSON can be 1k+ tokens
                }
            },
            timeout=300,   # 5 min — larger models + large output
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except Exception as e:
        print(f"  [!] Ollama error for {label}: {e}")
        return "{}"


# ── Shared JSON cleaner (used by autoscore_graph.py) ──────────────────────────

def strip_markdown_fences(raw: str) -> str:
    """
    Strips ```json ... ``` or ``` ... ``` wrappers if the model added them.
    Returns clean string ready for json.loads().
    """
    cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip()
    cleaned = cleaned.rstrip('`').strip()
    return cleaned
