# claim_pipeline/processing/ai_field_infer.py
"""
AI-backed field inference (simple, validator-compatible version).

This version is fully compatible with:
- normalizer.py
- validator.py
- decision_engine.py
- pipeline_runner.py

It returns the SAME structure normalizer uses:
{
    "original": str or None,
    "value": str or None,
    "success": bool,
    "notes": str
}

Supports:
- Groq LLM inference (strict JSON-only)
- Rule-based fallback
- Context-aware extraction
- Auto-normalization of inferred fields
"""

import os
import json
import re
import time
import hashlib
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

from claim_pipeline.processing.normalizer import (
    normalize_amount,
    normalize_date,
    normalize_phone,
    normalize_email,
    normalize_string,
)

load_dotenv()

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMP = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAXTOK = int(os.getenv("LLM_MAX_TOKENS", "800"))
AI_RETRIES = 2

OFFLINE_MODE = not bool(GROQ_API_KEY)

EXPECTED_FIELDS = [
    "claim_id",
    "policy_no",
    "claimant_name",
    "phone",
    "email",
    "vehicle",
    "damage",
    "amount_estimated",
    "date",
    "insured_amount",
]

# -----------------------------------------------------
# Groq client (if available)
# -----------------------------------------------------
llm_client = None
if not OFFLINE_MODE:
    try:
        from groq import Groq
        llm_client = Groq(api_key=GROQ_API_KEY)
    except:
        llm_client = None
        OFFLINE_MODE = True


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def _hash_prompt(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _extract_json(text: str) -> Optional[dict]:
    """Safely extract JSON object from model output."""
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.I).strip()
    text = re.sub(r"^```", "", text).strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                block = text[start:i+1]
                try:
                    return json.loads(block)
                except:
                    try:
                        block2 = block.replace("'", '"')
                        block2 = re.sub(r",\s*([}\]])", r"\1", block2)
                        return json.loads(block2)
                    except:
                        return None
    return None


def _prepare_context(raw_text: str, max_lines=40) -> str:
    lines = []
    for ln in raw_text.splitlines():
        if ln.strip():
            lines.append(ln.strip())
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def _normalize_field(field: str, value: Any) -> Dict[str, Any]:
    """Apply correct normalizer for each field."""
    if value in (None, "", "null", "None"):
        return {"original": value, "value": "", "success": False, "notes": "empty"}

    value = str(value).strip()

    if field in ("amount_estimated", "insured_amount"):
        return normalize_amount(value)
    elif field == "date":
        return normalize_date(value)
    elif field == "phone":
        return normalize_phone(value)
    elif field == "email":
        return normalize_email(value)
    else:
        return normalize_string(value)


# -----------------------------------------------------
# RULE-BASED INFERENCE
# -----------------------------------------------------
def _rule_infer(field: str, text: str) -> Optional[str]:
    t = text

    if field == "claim_id":
        m = re.search(r"(CL[M]?[\/\-]?\w+)", t, re.I)
        return m.group(1) if m else None

    if field == "policy_no":
        m = re.search(r"(P[NO][L]?\/?\w+)", t, re.I)
        return m.group(1) if m else None

    if field == "claimant_name":
        m = re.search(r"(?:Name|Claimant)\s*:\s*([A-Za-z\s]{3,50})", t)
        return m.group(1).title() if m else None

    if field == "email":
        m = re.search(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z0-9]{2,}", t)
        return m.group(0) if m else None

    if field == "phone":
        m = re.search(r"\+?\d[\d\s-]{7,}\d", t)
        return re.sub(r"[^\d+]", "", m.group(0)) if m else None

    if field == "date":
        m = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", t)
        return m.group(1) if m else None

    if field in ("amount_estimated", "insured_amount"):
        m = re.search(r"(₹\s?[0-9,\.]+|Rs\.?\s?[0-9,\.]+|\$[0-9,\.]+)", t, re.I)
        return m.group(1) if m else None

    if field == "vehicle":
        m = re.search(r"(?:Model|Make|Vehicle)\s*:\s*(.*)", t)
        return m.group(1).strip() if m else None

    if field == "damage":
        m = re.search(r"(?:Nature of Damage|Damage)\s*:\s*(.*)", t)
        return m.group(1).strip() if m else None

    return None


# -----------------------------------------------------
# LLM CALL
# -----------------------------------------------------
def _call_groq(prompt: str) -> Optional[dict]:
    if OFFLINE_MODE or not llm_client:
        return None

    for _ in range(AI_RETRIES):
        try:
            resp = llm_client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMP,
                max_tokens=LLM_MAXTOK,
                messages=[
                    {"role": "system", "content": "Return ONLY JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            out = resp.choices[0].message.content
            return _extract_json(out)
        except:
            time.sleep(0.8)

    return None


# -----------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------
def infer_claim_fields(raw_text: str,
                       extracted_fields: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Returns normalized fields for those missing from extraction.
    Keeps structure identical to normalizer output.
    """

    context = _prepare_context(raw_text)

    # fields that need inference
    missing = []
    for f in EXPECTED_FIELDS:
        ef = extracted_fields.get(f, {})
        if not isinstance(ef, dict) or not ef.get("value"):
            missing.append(f)

    if not missing:
        return extracted_fields  # nothing to infer

    # build prompt
    prompt = (
        "Extract missing insurance claim fields from text.\n"
        "Return STRICT JSON.\n"
        "Keys required: " + ", ".join(missing) + "\n\n"
        "TEXT:\n" + context
    )

    # Try Groq
    parsed = _call_groq(prompt)

    final = extracted_fields.copy()

    for f in missing:
        llm_val = None
        if parsed and f in parsed:
            llm_val = parsed[f]

        if not llm_val:
            llm_val = _rule_infer(f, raw_text)

        final[f] = _normalize_field(f, llm_val)

    return final

