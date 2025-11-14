# claim_pipeline/processing/ai_field_infer.py
"""
AI-backed field inference using Groq OR rule fallback.
Fully patched version — 100% test suite compatible.

Major guarantees:
- ALWAYS returns all EXPECTED_FIELDS
- ALWAYS returns `_ai_change_log`
- Rule-mode now performs full post-processing exactly like LLM mode
- Cache stores fully shaped results
"""

from __future__ import annotations
import os
import json
import re
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

# Structured normalizers for post-verification
from claim_pipeline.processing.normalizer import (
    normalize_amount,
    normalize_date,
    normalize_phone,
    normalize_email,
    normalize_string,
)

load_dotenv()

# -----------------------
# Config
# -----------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
RETRY_COUNT = int(os.getenv("AI_RETRY_COUNT", "3"))
RETRY_BACKOFF = float(os.getenv("AI_RETRY_BACKOFF", "1.0"))
CACHE_ENABLED = os.getenv("AI_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
OFFLINE_STUB = os.getenv("AI_OFFLINE_STUB", "false").lower() in ("1", "true", "yes")
LOG_LEVEL = os.getenv("AI_LOG_LEVEL", "INFO")

EXPECTED_FIELDS = [
    "claim_id", "policy_no", "claimant_name", "phone", "email",
    "vehicle", "damage", "amount_estimated", "date", "insured_amount"
]

logger = logging.getLogger("ai_field_infer")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)
    fmt = logging.Formatter("[ai_field_infer] %(levelname)s: %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)


# -----------------------
# Groq client init
# -----------------------
llm_client = None
llm_available = False
if not OFFLINE_STUB and GROQ_API_KEY:
    try:
        from groq import Groq
        llm_client = Groq(api_key=GROQ_API_KEY)
        llm_available = True
        logger.info("Groq client initialized.")
    except Exception as e:
        logger.warning(f"Groq init failed: {e}")
        llm_client = None
else:
    logger.info("Groq disabled — using offline stub / rule-based inference.")


# -----------------------
# Cache
# -----------------------
_cache: Dict[str, Dict[str, Any]] = {}

def _cache_get(k): return _cache.get(k) if CACHE_ENABLED else None
def _cache_set(k, v): 
    if CACHE_ENABLED: 
        _cache[k] = v


# -----------------------
# HELPERS
# -----------------------
def _hash_prompt(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _prepare_context(raw_text: str, partial_fields: Dict[str, Any], max_lines=40):
    lines = []

    if isinstance(partial_fields, dict) and "raw_lines" in partial_fields:
        try:
            for ln in partial_fields["raw_lines"][:max_lines]:
                text = ln.get("text", "")
                if text:
                    lines.append(text)
        except:
            pass

    for f in EXPECTED_FIELDS:
        pf = partial_fields.get(f)
        if isinstance(pf, dict):
            if pf.get("raw_candidates"):
                for rc in pf["raw_candidates"]:
                    if rc not in lines:
                        lines.append(str(rc))
            if pf.get("original"):
                s = str(pf["original"])
                if s not in lines:
                    lines.append(s)

    if not lines:
        for ln in raw_text.splitlines():
            if ln.strip():
                lines.append(ln.strip())
            if len(lines) >= max_lines:
                break

    return "\n".join(lines[:max_lines])


_FEW_SHOT_EXAMPLES = [
    {
        "text": "Claim ID: CLM-100\nPolicy: PN-2222\nDate: 2024-10-12",
        "json": {
            "claim_id": {"value": "CLM-100", "confidence": 0.95, "explain": "simple extraction"},
            "policy_no": {"value": "PN-2222", "confidence": 0.9, "explain": ""},
            "date": {"value": "2024-10-12", "confidence": 0.9, "explain": ""},
        }
    }
]


def _build_prompt_for_fields(context, partial_fields, fields_to_infer):
    compact = {
        f: {
            "value": partial_fields.get(f, {}).get("value"),
            "score": partial_fields.get(f, {}).get("score"),
            "notes": partial_fields.get(f, {}).get("notes", [])[:2],
        }
        for f in fields_to_infer
    }

    examples = ""
    for ex in _FEW_SHOT_EXAMPLES:
        examples += f"TEXT:\n{ex['text']}\nEXPECTED_JSON:\n{json.dumps(ex['json'], indent=2)}\n\n"

    prompt = (
        "Return ONLY JSON.\n"
        "JSON must contain keys: value, confidence, explain.\n\n"
        "FEW-SHOT:\n" + examples +
        "CONTEXT:\n" + context + "\n\n"
        "PARTIAL_FIELDS:\n" + json.dumps(compact, indent=2) + "\n\n"
        "INFER THESE FIELDS:\n" + json.dumps(fields_to_infer) + "\n"
    )
    return prompt


def _extract_json_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    text = re.sub(r"^```json", "", text.strip(), flags=re.I)
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
                return text[start:i + 1]
    return None


def _safe_json_load(s: str):
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


def _call_groq_chat(prompt: str):
    if not llm_available or not llm_client:
        return None

    for attempt in range(RETRY_COUNT):
        try:
            resp = llm_client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": "Return ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            logger.warning(f"Groq attempt {attempt+1} failed: {e}. Sleeping {wait}s.")
            time.sleep(wait)
    return None


# -----------------------
# Verification
# -----------------------
def _verify_and_shape_field(field: str, llm_val, llm_conf, partial_pf):
    shaped = {
        "original": llm_val,
        "value": None,
        "canonical": None,
        "confidence": "low",
        "score": float(llm_conf or 0.0),
        "corrected": False,
        "notes": [],
        "metadata": {},
        "source": "llm",
        "explain": "",
    }

    if llm_val in (None, "", "null"):
        shaped["notes"].append("llm_empty")
        return shaped

    try:
        if field in ("amount_estimated", "insured_amount"):
            norm = normalize_amount(str(llm_val))
        elif field == "date":
            norm = normalize_date(str(llm_val))
        elif field == "phone":
            norm = normalize_phone(str(llm_val))
        elif field == "email":
            norm = normalize_email(str(llm_val))
        else:
            norm = normalize_string(str(llm_val))
    except:
        norm = None

    if isinstance(norm, dict):
        shaped["value"] = norm.get("value")
        shaped["canonical"] = norm.get("canonical")
        shaped["notes"] += norm.get("notes", [])
        shaped["metadata"].update(norm.get("metadata", {}))
        combined = 0.6 * norm.get("score", 0.0) + 0.4 * shaped["score"]
        shaped["score"] = combined
        shaped["confidence"] = "high" if combined >= 0.85 else ("medium" if combined >= 0.5 else "low")
        shaped["corrected"] = norm.get("corrected", False)
    else:
        shaped["value"] = llm_val

    return shaped


# -----------------------
# Rule-based fallback
# -----------------------
def _rule_infer_field(field: str, raw_text: str):
    t = raw_text or ""

    if field == "claim_id":
        m = re.search(r"(CLM[-\s:]?\d{3,})", t, re.I)
        return m.group(1).strip() if m else None

    if field == "policy_no":
        m = re.search(r"(PN[-\s:]?\d{3,})", t, re.I)
        return m.group(1).strip() if m else None

    if field == "claimant_name":
        m = re.search(r"(?:Name|Claimant)\s*[:\-]\s*([A-Za-z\s]{3,50})", t, re.I)
        return m.group(1).strip().title() if m else None

    if field == "amount_estimated":
        m = re.search(r"(₹\s?[0-9,\.]+|Rs\s?[0-9,\.]+|\$[0-9,\.]+)", t, re.I)
        return m.group(1).strip() if m else None

    if field == "date":
        m = re.search(r"(\d{4}-\d{2}-\d{2})", t)
        return m.group(1) if m else None

    if field == "email":
        m = re.search(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z0-9]{2,}", t)
        return m.group(0).lower() if m else None

    if field == "phone":
        m = re.search(r"\+?\d[\d\s\-]{7,}\d", t)
        return re.sub(r"[^\d+]", "", m.group(0)) if m else None

    if field == "vehicle":
        m = re.search(r"(?:Vehicle|Model)\s*[:\-]\s*([A-Za-z0-9\s\-\/]+)", t, re.I)
        return m.group(1).strip() if m else None

    if field == "damage":
        m = re.search(r"(Damage|Loss)[:\-]\s*(.{5,200})", t, re.I)
        return m.group(2).strip() if m else None

    return None


def _enhanced_rule_based_inference(raw_text, partial_fields, fields_to_infer):
    result = {}

    for f in fields_to_infer:
        pf = partial_fields.get(f)
        if isinstance(pf, dict) and pf.get("value") is not None:
            result[f] = {
                "original": pf.get("original"),
                "value": pf.get("value"),
                "canonical": pf.get("canonical"),
                "confidence": pf.get("confidence", "medium"),
                "score": pf.get("score", 0.6),
                "corrected": pf.get("corrected", False),
                "notes": pf.get("notes", []),
                "metadata": pf.get("metadata", {}),
                "source": "partial",
                "explain": "kept partial value",
            }
        else:
            inferred = _rule_infer_field(f, raw_text)
            shaped = _verify_and_shape_field(f, inferred, 0.35, pf)
            shaped["source"] = "rule"
            shaped["explain"] = f"inferred by rule heuristic for {f}" if inferred else ""
            result[f] = shaped

    return result


# -----------------------
# MAIN FUNCTION (patched)
# -----------------------
def infer_claim_fields(raw_text: str, partial_fields: Dict[str, Any],
                       fields_to_infer: Optional[List[str]] = None,
                       use_groq: Optional[bool] = None):

    logger.info(f"Starting inference for text len={len(raw_text)}")

    if fields_to_infer is None:
        fields_to_infer = []
        for f in EXPECTED_FIELDS:
            pf = partial_fields.get(f)
            if pf is None or (isinstance(pf, dict) and pf.get("value") is None):
                fields_to_infer.append(f)
            elif isinstance(pf, dict) and pf.get("score", 1.0) < 0.5:
                fields_to_infer.append(f)

    if use_groq is None:
        use_groq = llm_available

    context = _prepare_context(raw_text, partial_fields)
    prompt = _build_prompt_for_fields(context, partial_fields, fields_to_infer)
    prompt_hash = _hash_prompt(prompt)

    cached = _cache_get(prompt_hash)
    if cached:
        logger.info("Cache hit")
        return cached

    # -------------------------
    # RULE-BASED PATH (PATCHED)
    # -------------------------
    if OFFLINE_STUB or not use_groq:
        logger.info("Using rule-based inference")
        inferred = _enhanced_rule_based_inference(raw_text, partial_fields, fields_to_infer)

        # --- PATCH: Produce full shaped result identical to LLM path ---
        shaped = {}

        for f in EXPECTED_FIELDS:
            if f in inferred:
                shaped[f] = inferred[f]
            else:
                pf = partial_fields.get(f)
                if isinstance(pf, dict):
                    shaped[f] = {
                        "original": pf.get("original"),
                        "value": pf.get("value"),
                        "canonical": pf.get("canonical"),
                        "confidence": pf.get("confidence", "medium"),
                        "score": pf.get("score", 0.7),
                        "corrected": pf.get("corrected", False),
                        "notes": pf.get("notes", []),
                        "metadata": pf.get("metadata", {}),
                        "source": "partial",
                        "explain": "kept existing partial",
                    }
                else:
                    shaped[f] = {
                        "original": pf,
                        "value": pf,
                        "canonical": pf,
                        "confidence": "low" if pf is None else "medium",
                        "score": 0.0 if pf is None else 0.6,
                        "corrected": False,
                        "notes": [],
                        "metadata": {},
                        "source": "partial",
                        "explain": "",
                    }

        # --- PATCH: always include change log ---
        shaped["_ai_change_log"] = []

        _cache_set(prompt_hash, shaped)
        return shaped

    # -------------------------
    # LLM INFERENCE PATH
    # -------------------------
    llm_text = _call_groq_chat(prompt)
    if not llm_text:
        logger.warning("Groq failed → using rule-based fallback")
        return infer_claim_fields(raw_text, partial_fields, fields_to_infer, use_groq=False)

    json_block = _extract_json_from_text(llm_text)
    parsed = _safe_json_load(json_block or llm_text)

    if not parsed:
        logger.warning("Invalid JSON → fallback to rule-based")
        return infer_claim_fields(raw_text, partial_fields, fields_to_infer, use_groq=False)

    shaped = {}
    change_log = []

    for f in fields_to_infer:
        entry = parsed.get(f, {})
        llm_val = entry.get("value")
        llm_conf = entry.get("confidence", 0.0)
        explain = entry.get("explain", "")

        pf = partial_fields.get(f)
        shaped_field = _verify_and_shape_field(f, llm_val, llm_conf, pf)
        shaped_field["explain"] = explain

        shaped[f] = shaped_field
        change_log.append({"field": f, "new": shaped_field["value"], "rule": "llm_inferred"})

    # Fill remaining EXPECTED_FIELDS from partial
    for f in EXPECTED_FIELDS:
        if f not in shaped:
            pf = partial_fields.get(f)
            if isinstance(pf, dict):
                shaped[f] = {
                    "original": pf.get("original"),
                    "value": pf.get("value"),
                    "canonical": pf.get("canonical"),
                    "confidence": pf.get("confidence", "medium"),
                    "score": pf.get("score", 0.7),
                    "corrected": pf.get("corrected", False),
                    "notes": pf.get("notes", []),
                    "metadata": pf.get("metadata", {}),
                    "source": "partial",
                    "explain": "kept existing partial",
                }
            else:
                shaped[f] = {
                    "original": pf,
                    "value": pf,
                    "canonical": pf,
                    "confidence": "low" if pf is None else "medium",
                    "score": 0.0 if pf is None else 0.6,
                    "corrected": False,
                    "notes": [],
                    "metadata": {},
                    "source": "partial",
                    "explain": "",
                }

    shaped["_ai_change_log"] = change_log

    _cache_set(prompt_hash, shaped)
    return shaped












# # claim_pipeline/processing/ai_field_infer.py
# """
# AI-backed field inference using Groq Chat completions.

# Features implemented (full list of improvements):
# - Groq-only implementation (requires GROQ_API_KEY)
# - Structured input handling: accepts partial_fields as structured normalizer output
# - Prompt engineering with strict JSON schema + few-shot examples
# - Chunking: uses only evidence lines / top context to limit tokens
# - Retries with exponential backoff for transient API errors
# - Robust JSON extraction & validation, attempts basic repairs
# - Post-verification: runs local normalizers on LLM outputs (sanity checks)
# - Confidence-merging strategy (LLM vs partial)
# - Returns fully structured fields matching normalizer output shape:
#     { field: { "original": ..., "value": ..., "canonical": ..., "confidence": "...",
#                "score": float, "corrected": bool, "notes": [...], "metadata": {...},
#                "source": "llm"|"partial"|"rule", "explain": "one-sentence reason" } }
# - ai_change_log: what LLM changed vs partial_fields (for audit/UI)
# - ai_fields selection: call LLM only for fields listed (reduces tokens)
# - Offline stub mode for local dev (GROQ not required)
# - Logging (python logging)
# - Caching (simple in-memory cache based on prompt hash) to avoid repeated LLM calls

# Usage:
#     infer_claim_fields(raw_text, partial_fields, fields_to_infer=None, use_groq=True)

# Notes:
# - This module imports and uses the structured normalizer functions from processing.normalizer
#   to validate and canonicalize LLM outputs.
# - Requires: python-dotenv, groq (client), dateutil, etc. In dev mode, it will run rule-based fallback.
# """

# from __future__ import annotations
# import os
# import json
# import re
# import time
# import hashlib
# import logging
# from typing import Dict, Any, List, Optional, Tuple

# from dotenv import load_dotenv

# # Structured normalizers for post-verification
# from claim_pipeline.processing.normalizer import (
#     normalize_amount,
#     normalize_date,
#     normalize_phone,
#     normalize_email,
#     normalize_string,
# )

# load_dotenv()

# # -----------------------
# # Configuration
# # -----------------------
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
# LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
# LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
# RETRY_COUNT = int(os.getenv("AI_RETRY_COUNT", "3"))
# RETRY_BACKOFF = float(os.getenv("AI_RETRY_BACKOFF", "1.0"))  # base seconds
# CACHE_ENABLED = os.getenv("AI_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
# OFFLINE_STUB = os.getenv("AI_OFFLINE_STUB", "false").lower() in ("1", "true", "yes")
# LOG_LEVEL = os.getenv("AI_LOG_LEVEL", "INFO")

# # Fields we expect
# EXPECTED_FIELDS = [
#     "claim_id", "policy_no", "claimant_name", "phone", "email",
#     "vehicle", "damage", "amount_estimated", "date", "insured_amount"
# ]

# # Configure logger
# logger = logging.getLogger("ai_field_infer")
# logger.setLevel(LOG_LEVEL)
# if not logger.handlers:
#     ch = logging.StreamHandler()
#     ch.setLevel(LOG_LEVEL)
#     formatter = logging.Formatter("[ai_field_infer] %(levelname)s: %(message)s")
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

# # -----------------------
# # Groq client init (Groq-only)
# # -----------------------
# llm_client = None
# llm_available = False
# if not OFFLINE_STUB and GROQ_API_KEY:
#     try:
#         from groq import Groq
#         llm_client = Groq(api_key=GROQ_API_KEY)
#         llm_available = True
#         logger.info("Groq client initialized.")
#     except Exception as e:
#         logger.warning(f"Failed to initialize Groq client: {e}. Falling back to offline mode.")
#         llm_client = None
#         llm_available = False
# else:
#     logger.info("Groq client not configured or offline-stub enabled; using rule-based mode.")


# # -----------------------
# # Simple in-memory cache (prompt_hash -> parsed JSON)
# # -----------------------
# _cache: Dict[str, Dict[str, Any]] = {}


# def _cache_get(key: str) -> Optional[Dict[str, Any]]:
#     if not CACHE_ENABLED:
#         return None
#     return _cache.get(key)


# def _cache_set(key: str, value: Dict[str, Any]):
#     if not CACHE_ENABLED:
#         return
#     _cache[key] = value


# # -----------------------
# # Helpers: prompt, chunking, JSON extraction, validation
# # -----------------------
# def _hash_prompt(prompt: str) -> str:
#     return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# def _prepare_context(raw_text: str, partial_fields: Dict[str, Any], max_lines: int = 40) -> str:
#     """
#     Build a compact context for the LLM: prefer evidence lines from partial_fields if available,
#     else use the first/most relevant lines from raw_text. Limit to max_lines to control tokens.
#     """
#     # If partial_fields contain raw_lines evidence (as in extractor), use them
#     context_lines: List[str] = []
#     # partial_fields might contain 'raw_lines' at top-level (validator produced it)
#     if isinstance(partial_fields, dict) and "raw_lines" in partial_fields:
#         # raw_lines expected as list of {"index": int, "text": "..."}
#         try:
#             for ln in partial_fields["raw_lines"][:max_lines]:
#                 context_lines.append(ln.get("text", str(ln)))
#         except Exception:
#             pass

#     # Also allow partial_fields to include per-field raw_candidates
#     for fname in EXPECTED_FIELDS:
#         pf = partial_fields.get(fname)
#         if isinstance(pf, dict):
#             # if normalizer provided raw_candidates list, include them
#             raw_cands = pf.get("raw_candidates") or []
#             for rc in raw_cands:
#                 if rc and rc not in context_lines:
#                     context_lines.append(str(rc))
#             # include the normalized 'original' if exists
#             orig = pf.get("original")
#             if orig:
#                 s = str(orig)
#                 if s not in context_lines:
#                     context_lines.append(s)

#     # fallback: split raw_text lines, choose top lines
#     if not context_lines:
#         for ln in raw_text.splitlines():
#             if ln.strip():
#                 context_lines.append(ln.strip())
#             if len(context_lines) >= max_lines:
#                 break

#     # ensure we don't exceed max_lines
#     context_lines = context_lines[:max_lines]
#     context = "\n".join(context_lines)
#     return context


# _FEW_SHOT_EXAMPLES = [
#     {
#         "text": "Claim form: Claim ID: CLM-2024-001\nPolicy No: PN-7812\nName: John Doe\nAmount: ₹ 45,000\nDate: 2024-11-10",
#         "json": {
#             "claim_id": {"value": "CLM-2024-001", "confidence": 0.95, "explain": "explicit claim id line"},
#             "policy_no": {"value": "PN-7812", "confidence": 0.95, "explain": "explicit policy line"},
#             "claimant_name": {"value": "John Doe", "confidence": 0.9, "explain": "explicit name"},
#             "phone": {"value": None, "confidence": 0.0, "explain": ""},
#             "email": {"value": None, "confidence": 0.0, "explain": ""},
#             "vehicle": {"value": None, "confidence": 0.0, "explain": ""},
#             "damage": {"value": None, "confidence": 0.0, "explain": ""},
#             "amount_estimated": {"value": "₹ 45,000", "confidence": 0.95, "explain": "explicit amount"},
#             "date": {"value": "2024-11-10", "confidence": 0.95, "explain": "explicit date"},
#             "insured_amount": {"value": None, "confidence": 0.0, "explain": ""}
#         }
#     }
# ]


# def _build_prompt_for_fields(context: str, partial_fields: Dict[str, Any], fields_to_infer: List[str]) -> str:
#     """
#     Build a strict prompt asking Groq to return ONLY JSON following the schema.
#     """
#     # Compose partial_fields summary for the model (compact)
#     compact_partial = {}
#     for f in fields_to_infer:
#         pf = partial_fields.get(f)
#         if isinstance(pf, dict):
#             compact_partial[f] = {
#                 "value": pf.get("value"),
#                 "score": pf.get("score"),
#                 "confidence": pf.get("confidence"),
#                 "notes": pf.get("notes")[:3] if pf.get("notes") else []
#             }
#         else:
#             compact_partial[f] = pf

#     examples_text = ""
#     for ex in _FEW_SHOT_EXAMPLES[:1]:
#         examples_text += f"TEXT:\n{ex['text']}\nEXPECTED_JSON:\n{json.dumps(ex['json'], indent=2)}\n\n"

#     schema = {
#         "description": "Return a JSON object mapping each requested field to { value: string|null, confidence: number(0-1), explain: one-sentence reason }",
#         "fields": {f: {"type": ["string", "null"], "explain": "value or null"} for f in fields_to_infer}
#     }

#     prompt = (
#         "You are an assistant specialized in extracting structured claim fields from noisy OCR/text. "
#         "You must return ONLY a JSON object and nothing else. The JSON object should contain exactly the keys requested "
#         f"and each key must be an object with fields: value (string or null), confidence (float 0.0-1.0), explain (one short sentence).\n\n"
#         "FEW-SHOT EXAMPLES:\n"
#         f"{examples_text}\n"
#         "CONTEXT (evidence lines):\n"
#         f"{context}\n\n"
#         "PARTIAL_FIELDS (what we already have):\n"
#         f"{json.dumps(compact_partial, indent=2, ensure_ascii=False)}\n\n"
#         "REQUEST: Infer values for the following fields: "
#         f"{json.dumps(fields_to_infer)}\n\n"
#         "Return ONLY the JSON object. No prose, no commentary, no code fences."
#     )
#     return prompt


# def _extract_json_from_text(text: str) -> Optional[str]:
#     """
#     Attempt to find the first JSON object in the text and return it.
#     This is robust to leading/trailing garbage and common markdown fences.
#     """
#     if not text:
#         return None
#     # strip code fences
#     text = text.strip()
#     # if starts with ```json remove it
#     text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
#     text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
#     # find first { ... } balanced block
#     start = text.find("{")
#     if start == -1:
#         return None
#     # attempt to find matching closing brace by scanning
#     depth = 0
#     for i in range(start, len(text)):
#         ch = text[i]
#         if ch == "{":
#             depth += 1
#         elif ch == "}":
#             depth -= 1
#             if depth == 0:
#                 candidate = text[start:i + 1]
#                 return candidate
#     # fallback: try to strip non-json and load direct
#     return None


# def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
#     """
#     Try to fix trivial JSON issues (single quotes, trailing commas) then json.loads.
#     """
#     if not s:
#         return None
#     try:
#         return json.loads(s)
#     except json.JSONDecodeError:
#         # simple fixes
#         s2 = s.strip()
#         # replace single quotes to double quotes (careful)
#         s2 = re.sub(r"'", '"', s2)
#         # remove trailing commas before closing braces/brackets
#         s2 = re.sub(r",\s*([}\]])", r"\1", s2)
#         try:
#             return json.loads(s2)
#         except Exception:
#             return None


# # -----------------------
# # Groq call with retries
# # -----------------------
# def _call_groq_chat(prompt: str, timeout: int = 30) -> Optional[str]:
#     """
#     Call Groq chat completions with retries and exponential backoff.
#     Returns the model text or None on failure.
#     """
#     if not llm_available or not llm_client:
#         logger.debug("Groq client not available.")
#         return None

#     attempt = 0
#     while attempt < RETRY_COUNT:
#         try:
#             start_ts = time.time()
#             response = llm_client.chat.completions.create(
#                 model=LLM_MODEL,
#                 messages=[
#                     {"role": "system", "content": "You are a strict JSON-output claims field extractor. Return only JSON."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=LLM_TEMPERATURE,
#                 max_tokens=LLM_MAX_TOKENS
#             )
#             # Groq response structure: response.choices[0].message.content (aligns with previous usage)
#             out = response.choices[0].message.content
#             latency = time.time() - start_ts
#             logger.info(f"Groq call success (attempt={attempt+1}) latency={latency:.2f}s")
#             return out
#         except Exception as e:
#             attempt += 1
#             wait = RETRY_BACKOFF * (2 ** (attempt - 1))
#             logger.warning(f"Groq call failed (attempt={attempt}/{RETRY_COUNT}): {e}. Backing off {wait:.1f}s.")
#             time.sleep(wait)
#     logger.error("Groq call failed after retries.")
#     return None


# # -----------------------
# # Post-processing & verification
# # -----------------------
# def _verify_and_shape_field(field: str, llm_val: Any, llm_conf: float, partial_pf: Any) -> Dict[str, Any]:
#     """
#     Verify LLM-proposed value using local normalizers and return a fully shaped dict
#     consistent with normalizer output.
#     """
#     shaped = {
#         "original": llm_val,
#         "value": None,
#         "canonical": None,
#         "confidence": "low",
#         "score": 0.0,
#         "corrected": False,
#         "notes": [],
#         "metadata": {},
#         "source": "llm",
#         "explain": ""
#     }

#     # basic fill
#     if llm_val in (None, "", "null"):
#         shaped["value"] = None
#         shaped["score"] = float(llm_conf or 0.0)
#         shaped["confidence"] = _score_label(shaped["score"])
#         shaped["notes"].append("llm_proposed_empty")
#         return shaped

#     # convert llm_conf to 0-1 float safely
#     try:
#         conf = float(llm_conf)
#     except Exception:
#         conf = 0.0
#     shaped["score"] = max(0.0, min(1.0, conf))
#     shaped["confidence"] = _score_label(shaped["score"])

#     # call relevant normalizer for sanity checking & canonicalization
#     try:
#         if field in ("amount_estimated", "insured_amount"):
#             norm = normalize_amount(str(llm_val))
#         elif field == "date":
#             norm = normalize_date(str(llm_val))
#         elif field == "phone":
#             norm = normalize_phone(str(llm_val))
#         elif field == "email":
#             norm = normalize_email(str(llm_val))
#         else:
#             norm = normalize_string(str(llm_val))
#     except Exception as e:
#         logger.warning(f"Normalizer failed for field={field} on llm_val={llm_val}: {e}")
#         norm = None

#     # If normalizer returned structured dict, use it to improve canonical/value fields
#     if isinstance(norm, dict):
#         # adopt normalized canonical/value when present
#         shaped["value"] = norm.get("value")
#         shaped["canonical"] = norm.get("canonical")
#         # merge notes & metadata
#         shaped["notes"].extend(norm.get("notes") or [])
#         shaped["metadata"].update(norm.get("metadata") or {})
#         # adjust score/confidence slightly by combining LLM and normalizer
#         normalizer_score = norm.get("score", 0.0)
#         # combine scores: weighted average favoring normalizer for syntactic checks
#         combined_score = (0.6 * normalizer_score) + (0.4 * shaped["score"])
#         shaped["score"] = max(0.0, min(1.0, combined_score))
#         shaped["confidence"] = _score_label(shaped["score"])
#         # indicate if correction happened
#         shaped["corrected"] = bool(norm.get("corrected", False))
#     else:
#         # no normalizer output available: keep raw
#         shaped["value"] = llm_val

#     # source attribution: if partial_pf exists and equals value, prefer 'partial'
#     if isinstance(partial_pf, dict) and partial_pf.get("value") and partial_pf.get("value") == shaped["value"]:
#         shaped["source"] = "partial"
#         # merge partial's notes/scores
#         pscore = partial_pf.get("score", 0.0)
#         shaped["score"] = max(shaped["score"], pscore)
#         shaped["confidence"] = _score_label(shaped["score"])

#     return shaped


# def _score_label(score: float) -> str:
#     if score >= 0.85:
#         return "high"
#     if score >= 0.5:
#         return "medium"
#     return "low"


# # -----------------------
# # Rule-based fallback (enhanced)
# # -----------------------
# def _rule_infer_field(field: str, raw_text: str) -> Optional[str]:
#     """
#     Lightweight heuristics to infer common fields from text.
#     This is intentionally conservative and low-confidence.
#     """
#     t = raw_text or ""
#     t_lower = t.lower()

#     if field == "claim_id":
#         m = re.search(r"(clm[-\s:]?[:\s]*[A-z0-9\-]{3,})", t, re.IGNORECASE)
#         if m:
#             return m.group(1).strip()
#     if field == "policy_no":
#         m = re.search(r"(pn[-\s:]?[:\s]*[A-z0-9\-]{3,})", t, re.IGNORECASE)
#         if m:
#             return m.group(1).strip()
#     if field == "claimant_name":
#         m = re.search(r"(?:name|claimant|insured)[:\s]+([A-Za-z][A-Za-z\s]{2,50})", t, re.IGNORECASE)
#         if m:
#             return m.group(1).strip().title()
#     if field == "amount_estimated":
#         m = re.search(r"(?:total|amount|estimate)[^\d₹$]{0,10}([₹$]?\s?[0-9,\.]+)", t, re.IGNORECASE)
#         if m:
#             return m.group(1).strip()
#     if field == "date":
#         m = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})", t)
#         if m:
#             return m.group(1).strip()
#     if field == "email":
#         m = re.search(r"([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[A-Za-z0-9.\-]+)", t)
#         if m:
#             return m.group(1).strip().lower()
#     if field == "phone":
#         m = re.search(r"(\+?\d[\d\-\s]{6,}\d)", t)
#         if m:
#             return re.sub(r"[^\d\+]", "", m.group(1).strip())
#     if field == "vehicle":
#         m = re.search(r"(vehicle|car|model|make)[:\s]+([A-Za-z0-9\s\-\/]+)", t, re.IGNORECASE)
#         if m:
#             return m.group(2).strip()
#     if field == "damage":
#         m = re.search(r"(damage|loss|damage description)[:\s]+(.{5,200})", t, re.IGNORECASE)
#         if m:
#             return m.group(2).strip()
#     return None


# def _enhanced_rule_based_inference(raw_text: str, partial_fields: Dict[str, Any], fields_to_infer: List[str]) -> Dict[str, Dict[str, Any]]:
#     """
#     Conservative rule-based fallback that produces shaped outputs similar to LLM path.
#     """
#     logger.info("Using enhanced rule-based inference")
#     result: Dict[str, Dict[str, Any]] = {}
#     for f in fields_to_infer:
#         partial_pf = partial_fields.get(f) if isinstance(partial_fields, dict) else None
#         if isinstance(partial_pf, dict) and partial_pf.get("value") is not None:
#             # keep partial as-is, but ensure structure
#             result[f] = {
#                 "original": partial_pf.get("original"),
#                 "value": partial_pf.get("value"),
#                 "canonical": partial_pf.get("canonical"),
#                 "confidence": partial_pf.get("confidence", "medium"),
#                 "score": partial_pf.get("score", 0.7),
#                 "corrected": partial_pf.get("corrected", False),
#                 "notes": partial_pf.get("notes", []),
#                 "metadata": partial_pf.get("metadata", {}),
#                 "source": "partial",
#                 "explain": "kept existing partial value"
#             }
#         else:
#             inferred = _rule_infer_field(f, raw_text)
#             shaped = _verify_and_shape_field(f, inferred, 0.35, partial_pf)
#             # Mark source as rule if we inferred it here
#             shaped["source"] = "rule" if inferred else "rule"
#             if inferred:
#                 shaped["explain"] = f"inferred by rule heuristic for {f}"
#             else:
#                 shaped["explain"] = ""
#             result[f] = shaped
#     return result


# # -----------------------
# # Top-level function
# # -----------------------
# def infer_claim_fields(raw_text: str, partial_fields: Dict[str, Any],
#                        fields_to_infer: Optional[List[str]] = None,
#                        use_groq: Optional[bool] = None) -> Dict[str, Dict[str, Any]]:
#     """
#     Infer missing/low-confidence claim fields.

#     Args:
#         raw_text: full extracted text (string)
#         partial_fields: dict of structured normalizer outputs for fields (may include 'raw_lines')
#         fields_to_infer: list of fields to ask LLM for (defaults to fields with missing/low confidence)
#         use_groq: override: True to attempt Groq, False to force rule-based

#     Returns:
#         dict mapping field -> shaped structured dict (compatible with normalizer output)
#     """
#     logger.info(f"Starting inference for text len={len(raw_text)} partial_fields_keys={list(partial_fields.keys()) if isinstance(partial_fields, dict) else 'N/A'}")

#     # Determine which fields to infer if not provided
#     if fields_to_infer is None:
#         # infer fields that are missing or low-confidence
#         fields_to_infer = []
#         for f in EXPECTED_FIELDS:
#             pf = partial_fields.get(f) if isinstance(partial_fields, dict) else None
#             if not pf or (isinstance(pf, dict) and pf.get("value") is None):
#                 fields_to_infer.append(f)
#             elif isinstance(pf, dict) and pf.get("score", 1.0) < 0.5:
#                 fields_to_infer.append(f)
#     logger.info(f"Fields to infer: {fields_to_infer}")

#     if not fields_to_infer:
#         logger.info("No fields require inference; returning partial_fields shaped.")
#         # shape partial_fields into expected shaped dicts if necessary
#         shaped = {}
#         for f in EXPECTED_FIELDS:
#             pf = partial_fields.get(f) if isinstance(partial_fields, dict) else None
#             if isinstance(pf, dict):
#                 shaped[f] = {
#                     "original": pf.get("original"),
#                     "value": pf.get("value"),
#                     "canonical": pf.get("canonical"),
#                     "confidence": pf.get("confidence", "medium"),
#                     "score": pf.get("score", 0.7),
#                     "corrected": pf.get("corrected", False),
#                     "notes": pf.get("notes", []),
#                     "metadata": pf.get("metadata", {}),
#                     "source": "partial",
#                     "explain": "kept existing partial value"
#                 }
#             else:
#                 shaped[f] = {
#                     "original": pf,
#                     "value": pf,
#                     "canonical": pf,
#                     "confidence": "low" if pf is None else "medium",
#                     "score": 0.0 if pf is None else 0.6,
#                     "corrected": False,
#                     "notes": [],
#                     "metadata": {},
#                     "source": "partial",
#                     "explain": ""
#                 }
#         return shaped

#     # Build context (chunking / evidence selection)
#     context = _prepare_context(raw_text, partial_fields, max_lines=40)
#     prompt = _build_prompt_for_fields(context, partial_fields, fields_to_infer)
#     prompt_hash = _hash_prompt(prompt)

#     # Check cache
#     cached = _cache_get(prompt_hash)
#     if cached:
#         logger.info("Cache hit for prompt; returning cached inference.")
#         return cached

#     # Determine whether to call Groq
#     if use_groq is None:
#         use_groq = llm_available

#     if OFFLINE_STUB or not use_groq:
#         # Skip calling Groq; use rule-based inference
#         result = _enhanced_rule_based_inference(raw_text, partial_fields, fields_to_infer)
#         _cache_set(prompt_hash, result)
#         return result

#     # Call Groq
#     llm_text = _call_groq_chat(prompt)
#     if not llm_text:
#         logger.warning("Groq returned no text; using rule-based fallback.")
#         result = _enhanced_rule_based_inference(raw_text, partial_fields, fields_to_infer)
#         _cache_set(prompt_hash, result)
#         return result

#     # Extract JSON from response robustly
#     json_block = _extract_json_from_text(llm_text)
#     parsed = _safe_json_load(json_block) if json_block else _safe_json_load(llm_text)
#     if not parsed:
#         logger.warning("Failed to parse LLM JSON output; attempting rule-based fallback.")
#         logger.debug(f"LLM raw output: {llm_text}")
#         result = _enhanced_rule_based_inference(raw_text, partial_fields, fields_to_infer)
#         _cache_set(prompt_hash, result)
#         return result

#     # Validate and shape parsed output
#     shaped_result: Dict[str, Dict[str, Any]] = {}
#     ai_change_log: List[Dict[str, Any]] = []
#     for f in fields_to_infer:
#         pf = partial_fields.get(f) if isinstance(partial_fields, dict) else None
#         entry = parsed.get(f)
#         if isinstance(entry, dict):
#             llm_val = entry.get("value")
#             llm_conf = entry.get("confidence", 0.0)
#             explanation = entry.get("explain", "")[:300] if entry.get("explain") else ""
#         else:
#             # if model returned direct string -> treat as value with low confidence
#             llm_val = entry
#             llm_conf = 0.5
#             explanation = ""

#         shaped = _verify_and_shape_field(f, llm_val, llm_conf, pf)
#         shaped["explain"] = explanation

#         # merging strategy: if partial exists and has high score, prefer partial unless LLM very confident
#         if isinstance(pf, dict) and pf.get("value") is not None:
#             pscore = float(pf.get("score", 0.0) or 0.0)
#             lscore = float(shaped.get("score", 0.0) or 0.0)
#             # override threshold: if LLM score > 0.75 and > pscore + 0.15, accept LLM
#             if lscore > 0.75 and (lscore - pscore) > 0.15:
#                 final = shaped
#                 final["source"] = "llm"
#                 ai_change_log.append({"field": f, "old": pf.get("value"), "new": final.get("value"),
#                                       "rule": "llm_override", "llm_score": lscore, "partial_score": pscore})
#             else:
#                 # keep partial, but update notes if LLM disagrees
#                 final = {
#                     "original": pf.get("original"),
#                     "value": pf.get("value"),
#                     "canonical": pf.get("canonical"),
#                     "confidence": pf.get("confidence", "medium"),
#                     "score": pf.get("score", 0.7),
#                     "corrected": pf.get("corrected", False),
#                     "notes": pf.get("notes", []),
#                     "metadata": pf.get("metadata", {}),
#                     "source": "partial",
#                     "explain": pf.get("notes", ["kept existing"])[0] if pf.get("notes") else "kept existing partial"
#                 }
#                 # if LLM proposed a different value, include it as note for human
#                 if shaped.get("value") != final["value"]:
#                     final["notes"].append(f"llm_suggested={shaped.get('value')};llm_score={shaped.get('score')}")
#                     ai_change_log.append({"field": f, "old": pf.get("value"), "new": shaped.get("value"),
#                                           "rule": "llm_suggested_keep_partial", "llm_score": shaped.get("score"),
#                                           "partial_score": pscore})
#         else:
#             # no partial available — accept shaped
#             final = shaped
#             final["source"] = "llm" if shaped.get("value") is not None else "rule"

#             if final["source"] == "llm":
#                 ai_change_log.append({"field": f, "old": None, "new": final.get("value"),
#                                       "rule": "llm_inferred", "llm_score": final.get("score")})

#         shaped_result[f] = final

#     # For fields not requested, populate from partial_fields (keep structure)
#     for f in EXPECTED_FIELDS:
#         if f not in shaped_result:
#             pf = partial_fields.get(f) if isinstance(partial_fields, dict) else None
#             if isinstance(pf, dict):
#                 shaped_result[f] = {
#                     "original": pf.get("original"),
#                     "value": pf.get("value"),
#                     "canonical": pf.get("canonical"),
#                     "confidence": pf.get("confidence", "medium"),
#                     "score": pf.get("score", 0.7),
#                     "corrected": pf.get("corrected", False),
#                     "notes": pf.get("notes", []),
#                     "metadata": pf.get("metadata", {}),
#                     "source": "partial",
#                     "explain": "kept existing partial"
#                 }
#             else:
#                 shaped_result[f] = {
#                     "original": pf,
#                     "value": pf,
#                     "canonical": pf,
#                     "confidence": "low" if pf is None else "medium",
#                     "score": 0.0 if pf is None else 0.6,
#                     "corrected": False,
#                     "notes": [],
#                     "metadata": {},
#                     "source": "partial",
#                     "explain": ""
#                 }

#     # attach change log
#     shaped_result["_ai_change_log"] = ai_change_log

#     # cache & return
#     _cache_set(prompt_hash, shaped_result)
#     logger.info("AI inference completed and cached.")
#     return shaped_result
