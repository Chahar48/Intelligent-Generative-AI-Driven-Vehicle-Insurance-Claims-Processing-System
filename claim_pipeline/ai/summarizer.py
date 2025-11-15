# claim_pipeline/summarizer/summarizer.py
"""
Clean, logging-free summarizer module.
Stable, type-safe, no logging, no console output.
"""

from __future__ import annotations
import os
import json
import re
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
MAX_CONTEXT_LINES = 40


# ============================================================
# SAFE FLOAT — global fix for "< not supported…" errors
# ============================================================
def safe_float(x, default=0.0):
    """Safely convert anything to float. Never crashes."""
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except:
        return float(default)


# ============================================================
# LLM loader
# ============================================================
llm_client = None
llm_available = False


def _get_llm():
    """Load Groq client only if API key exists."""
    global llm_client, llm_available

    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        llm_available = False
        llm_client = None
        return None, False

    if llm_client and "Fake" in llm_client.__class__.__name__:
        return llm_client, True

    try:
        from groq import Groq
        llm_client = Groq(api_key=key)
        llm_available = True
        return llm_client, True
    except:
        llm_available = False
        llm_client = None
        return None, False


# ============================================================
# JSON extraction helpers
# ============================================================
def _extract_json_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    t = text.strip()
    t = re.sub(r"^```json\s*", "", t, flags=re.I)
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)

    start = t.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]

    return None


def _safe_json_load(s: str) -> Optional[dict]:
    if not s:
        return None
    try:
        return json.loads(s)
    except:
        s2 = s.replace("'", '"')
        s2 = re.sub(r",\s*([}\]])", r"\1", s2)
        try:
            return json.loads(s2)
        except:
            return None


# ============================================================
# Evidence & Field compression
# ============================================================
def _make_evidence(raw: str) -> List[str]:
    out = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if ln:
            out.append(ln)
        if len(out) >= MAX_CONTEXT_LINES:
            break
    return out


def _structured_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize fields safely for summarizer."""
    out = {}
    for k, v in fields.items():
        if not isinstance(v, dict):
            out[k] = {"value": str(v), "canonical": str(v), "confidence": "low"}
        else:
            out[k] = {
                "value": v.get("value"),
                "canonical": v.get("canonical") or v.get("value"),
                "confidence": v.get("confidence", "low"),
            }
    return out


# ============================================================
# Prompt builder
# ============================================================
def _build_summary_prompt(raw: str, fields: dict, validation: dict) -> str:
    evidence = _make_evidence(raw)
    compact = _structured_fields(fields)

    issues = validation.get("issues", [])
    missing = validation.get("missing_required", [])
    rec = validation.get("recommendation", "review")

    return f"""
You are a senior insurance claims officer.
Return ONLY JSON.

=====================
EVIDENCE
=====================
{chr(10).join(evidence)}

=====================
FIELDS
=====================
{json.dumps(compact, indent=2)}

=====================
VALIDATION
=====================
Issues: {issues}
Missing Required: {missing}
Recommendation: {rec}

=====================
OUTPUT FORMAT
=====================
{{
  "summary": "3-5 lines",
  "decision": "approve|review|reject",
  "reasoning": "short reasoning",
  "confidence": 0.0
}}
"""


# ============================================================
# LLM call — no logging, silent
# ============================================================
def _call_groq(prompt: str) -> Optional[str]:
    client, ok = _get_llm()
    if not ok or not client:
        return None

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": prompt},
            ]
        )
        return resp.choices[0].message.content
    except:
        return None


# ============================================================
# Fallback summarizer — fully safe
# ============================================================
def _fallback(fields: dict, validation: dict) -> dict:

    def canon(k):
        f = fields.get(k)
        if isinstance(f, dict):
            return f.get("canonical") or f.get("value") or "Unknown"
        return f or "Unknown"

    claim_id = canon("claim_id")
    vehicle = canon("vehicle")
    amount = canon("amount_estimated")

    issues = validation.get("issues", [])
    missing = validation.get("missing_required", [])

    reason = []
    if issues:
        reason.append("Issues: " + ", ".join(issues))
    if missing:
        reason.append("Missing: " + ", ".join(missing))

    return {
        "summary": f"Claim {claim_id} for {vehicle}, estimated amount {amount}.",
        "decision": validation.get("recommendation", "review"),
        "reasoning": " ".join(reason) or "No issues.",
        "confidence": 0.60,
        "source": "rule"
    }


# ============================================================
# PUBLIC API
# ============================================================
def generate_claim_summary(raw_text: str,
                           fields: Dict[str, Any],
                           validation: Dict[str, Any]) -> Dict[str, Any]:

    if not raw_text or not isinstance(fields, dict):
        return _fallback(fields or {}, validation or {})

    prompt = _build_summary_prompt(raw_text, fields, validation)
    llm_text = _call_groq(prompt)

    # fallback if Groq unavailable
    if not llm_text:
        return _fallback(fields, validation)

    json_block = _extract_json_from_text(llm_text)
    parsed = _safe_json_load(json_block) if json_block else _safe_json_load(llm_text)

    if not parsed:
        return _fallback(fields, validation)

    parsed.setdefault("summary", "")
    parsed.setdefault("decision", validation.get("recommendation", "review"))
    parsed.setdefault("reasoning", "")
    parsed["confidence"] = safe_float(parsed.get("confidence"), 0.5)
    parsed["source"] = "llm"

    return parsed
