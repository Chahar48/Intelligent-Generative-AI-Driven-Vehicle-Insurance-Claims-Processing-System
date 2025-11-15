

###############################Not giving better Result##########################################################



# claim_pipeline/processing/decision_engine.py
"""
Tiny, crash-proof decision engine for the simplified claims pipeline.

Improved behavior:
- Prioritizes normalized/canonical fields from the extractor/normalizer.
- Ignores "FOR OFFICE USE" or "Approved Amount:" placeholders when extracting amounts from text.
- Falls back to summary/validation raw text only when structured fields are missing.
- Honors validation.recommendation when present (as a soft override).
- Returns a deterministic, stable output schema.
"""

from typing import Dict, Any, List, Optional
import re


# ---------- helpers ----------
def _get_field_dict(fields: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return a field dict (or simple wrapper) with consistent keys."""
    f = fields.get(name)
    if isinstance(f, dict):
        # ensure keys exist
        return {
            "value": f.get("value"),
            "canonical": f.get("canonical"),
            "original": f.get("original", f.get("value")),
            "success": f.get("success", True) if "success" in f else bool(f.get("value")),
            "notes": f.get("notes", None),
            "method": f.get("method", None),
        }
    else:
        return {"value": f, "canonical": f, "original": f, "success": bool(f)}


def _to_float_safe(s: Optional[Any]) -> Optional[float]:
    """Try to convert a string/number to float. Returns None on failure."""
    if s is None:
        return None
    try:
        # Accept already numeric types
        if isinstance(s, (int, float)):
            return float(s)
        st = str(s).strip()
        # clear common currency symbols and spaces/commas
        st = re.sub(r"[₹Rs\.\s,INR\$]+", "", st, flags=re.IGNORECASE)
        # if empty after cleaning, bail
        if st == "":
            return None
        return float(st)
    except Exception:
        return None


def _low_confidence_fields(fields: Dict[str, Any]) -> List[str]:
    """
    Return list of fields that were attempted but marked not-successful.
    (Normalization attempted but success==False.)
    """
    low = []
    for name, f in fields.items():
        if isinstance(f, dict):
            success = f.get("success", True)
            orig = f.get("original", "")
            if not success and orig not in (None, "", []):
                low.append(name)
    return low


def _strip_office_use(text: str) -> str:
    """
    Remove / truncate everything after 'FOR OFFICE USE' or similar headers
    to avoid reading placeholders such as 'Approved Amount:' in the footer.
    """
    if not text:
        return ""
    # patterns that denote the office-use area (case-insensitive)
    markers = [
        r"for office use only",
        r"for office use",
        r"office use only",
        r"approved amount[:\s]*$",
        r"approved amount[:\s]",
    ]
    txt = text
    for m in markers:
        # find marker position (ignore case)
        idx = re.search(m, txt, flags=re.IGNORECASE)
        if idx:
            pos = idx.start()
            txt = txt[:pos]
    return txt


def _find_amounts_in_text(text: str) -> List[float]:
    """
    Find plausible numeric amounts in supplied text (excluding office-use).
    Returns list of floats parsed from the text; empty list if none found.
    """
    if not text:
        return []
    t = _strip_office_use(text)
    # look for patterns like ₹45,000 or Rs. 45,000 or 45000.00 or $1,234
    pat = re.compile(
        r"(?:₹|Rs\.?|INR|\$)?\s*([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)",
        flags=re.IGNORECASE,
    )
    out: List[float] = []
    for m in pat.finditer(t):
        num = m.group(1)
        try:
            # strip commas/spaces then parse
            val = float(re.sub(r"[,\s]", "", num))
            out.append(val)
        except Exception:
            continue
    return out


# ---------- rule-based decision (improved) ----------
def rule_based_decision(fields: Dict[str, Any], validation: Optional[Dict[str, Any]] = None,
                        summary: Optional[Dict[str, Any]] = None, full_text: Optional[str] = None) -> Dict[str, Any]:
    """
    fields: normalized field dicts (as produced by extractor+normalizer)
    validation: optional validation result (may contain recommendation)
    summary: optional summary dict (LLM) - may include 'summary' string
    full_text: optional full raw text (if available) to search for amounts
    """

    reasons: List[str] = []
    flags: List[str] = []

    # --- pick amounts (priority: canonical -> value -> validation -> summary -> raw text) ---
    amt_val = None
    insured_val = None

    # helper to fetch candidate
    def _candidate_amount(field_name: str) -> Optional[float]:
        f = _get_field_raw(fields, field_name)
        # canonical first
        if f.get("canonical"):
            v = _to_float_safe(f.get("canonical"))
            if v is not None:
                return v
        # then value
        if f.get("value"):
            v = _to_float_safe(f.get("value"))
            if v is not None:
                return v
        # then original text
        if f.get("original"):
            v = _to_float_safe(f.get("original"))
            if v is not None:
                return v
        return None

    # small wrapper to create a consistent dict for a field
    def _get_field_raw(fields_dict: Dict[str, Any], name: str) -> Dict[str, Any]:
        f = fields_dict.get(name)
        if isinstance(f, dict):
            return {
                "value": f.get("value"),
                "canonical": f.get("canonical"),
                "original": f.get("original", f.get("value")),
                "success": f.get("success", bool(f.get("value"))),
            }
        else:
            return {"value": f, "canonical": f, "original": f, "success": bool(f)}

    # Use local wrapper
    _get_field_raw = _get_field_raw  # type: ignore

    # try direct extraction
    amt_val = _candidate_amount("amount_estimated")
    insured_val = _candidate_amount("insured_amount")

    # fallback: validation object may have normalized canonical values (common in pipeline)
    if (amt_val is None or insured_val is None) and isinstance(validation, dict):
        # some validation outputs store normalized values under different keys; try to read them safely
        try:
            vmap = validation.get("fields", {}) if isinstance(validation.get("fields", {}), dict) else {}
            if amt_val is None:
                cand = vmap.get("amount_estimated") or vmap.get("amount") or {}
                if isinstance(cand, dict):
                    amt_val = _to_float_safe(cand.get("value") or cand.get("canonical") or cand.get("original"))
                else:
                    amt_val = _to_float_safe(cand)
            if insured_val is None:
                cand2 = vmap.get("insured_amount") or {}
                if isinstance(cand2, dict):
                    insured_val = _to_float_safe(cand2.get("value") or cand2.get("canonical") or cand2.get("original"))
                else:
                    insured_val = _to_float_safe(cand2)
        except Exception:
            pass

    # fallback: summary text or raw_text search (but exclude office-use)
    if (amt_val is None or insured_val is None):
        search_text = ""
        if isinstance(summary, dict) and summary.get("summary"):
            search_text += "\n" + str(summary.get("summary"))
        if full_text:
            search_text += "\n" + str(full_text)
        # find numeric amounts in text (outside office-use)
        found_amounts = _find_amounts_in_text(search_text)
        if found_amounts:
            # heuristic: largest number -> insured amount, smaller -> estimated amount (if both missing)
            found_amounts = sorted(found_amounts, reverse=True)
            if insured_val is None and len(found_amounts) >= 1:
                insured_val = insured_val or float(found_amounts[0])
            if amt_val is None and len(found_amounts) >= 2:
                amt_val = amt_val or float(found_amounts[1])
            # if only one found and one of the fields missing, prefer smaller -> estimated
            if len(found_amounts) == 1:
                if amt_val is None and insured_val is None:
                    # can't tell for sure — treat single as amount_estimated (safer)
                    amt_val = float(found_amounts[0])

    # now we have numeric amt_val and insured_val possibly None
    # record evidence
    evidence = {"amount": amt_val, "insured_amount": insured_val, "low_confidence_fields": _low_confidence_fields(fields)}

    # --- early override: if validation has explicit recommendation, respect it (soft override) ---
    if isinstance(validation, dict):
        rec = validation.get("recommendation") or validation.get("recommendation", None)
        if isinstance(rec, str) and rec.lower() in ("approve", "reject", "review"):
            # if validator approved and we don't have a clear contradiction, honor it
            if rec.lower() == "approve":
                # ensure not an obvious contradiction (like claimed > insured)
                if amt_val is not None and insured_val is not None and amt_val > insured_val:
                    # contradiction -> continue normal flow (do not blindly approve)
                    pass
                else:
                    return {
                        "decision": "approve",
                        "confidence": 0.92,
                        "requires_human": False if not evidence["low_confidence_fields"] else True,
                        "reasons": ["validator_recommendation_approve"],
                        "evidence": evidence,
                        "flags": ["validator_approve"]
                    }
            elif rec.lower() == "reject":
                return {
                    "decision": "reject",
                    "confidence": 0.92,
                    "requires_human": False,
                    "reasons": ["validator_recommendation_reject"],
                    "evidence": evidence,
                    "flags": ["validator_reject"]
                }
            # if validator said 'review' we'll continue with normal logic but take note
            if rec.lower() == "review":
                # don't override; but add reason note later
                pass

    # --- core decisioning (conservative) ---
    reasons: List[str] = []
    flags: List[str] = []
    decision = "review"
    confidence = 0.6

    # missing both amounts -> review
    if amt_val is None and insured_val is None:
        decision = "review"
        confidence = 0.45
        reasons.append("amount_and_insured_missing")
        flags.append("missing_amounts")
    elif amt_val is None and insured_val is not None:
        # insured available but claimed amount missing -> review but less severe
        decision = "review"
        confidence = 0.55
        reasons.append("claimed_amount_missing")
        flags.append("missing_claim_amount")
    elif amt_val is not None and insured_val is None:
        # claimed amount present but insured not present -> review, require human
        decision = "review"
        confidence = 0.55
        reasons.append("insured_amount_missing")
        flags.append("missing_insured_amount")
    else:
        # both present -> numeric checks
        # non-positive claimed amount
        if amt_val <= 0:
            decision = "reject"
            confidence = 0.98
            reasons.append("non_positive_claim_amount")
            flags.append("invalid_amount")
        elif insured_val <= 0:
            decision = "review"
            confidence = 0.5
            reasons.append("invalid_insured_amount")
            flags.append("invalid_insured")
        elif amt_val > insured_val:
            decision = "reject"
            confidence = 0.98
            reasons.append("claimed_amount_exceeds_insured")
            flags.append("over_limit")
        else:
            ratio = amt_val / insured_val if insured_val > 0 else 0.0
            # if claim is very close to limit -> review
            if ratio >= 0.9:
                decision = "review"
                confidence = 0.75
                reasons.append("amount_close_to_insured_limit")
                flags.append("near_limit")
            else:
                # normal within-limit small claim -> approve
                decision = "approve"
                confidence = 0.9
                reasons.append("amount_within_insured_limit")
                flags.append("within_limit")

    # penalize if many low-confidence fields were reported by normalizer
    low_conf = evidence.get("low_confidence_fields", [])
    if low_conf:
        # drop confidence and require human if not already
        confidence = max(0.0, confidence - 0.2)
        if "low_confidence_fields_present" not in reasons:
            reasons.append("low_confidence_fields_present")
        if "low_confidence" not in flags:
            flags.append("low_confidence")

    # if validation recommended review earlier, keep a note
    if isinstance(validation, dict) and validation.get("recommendation") and validation.get("recommendation").lower() == "review":
        if "validator_suggests_review" not in reasons:
            reasons.append("validator_suggests_review")
        if "validator_review" not in flags:
            flags.append("validator_review")

    # ensure confidence bounds [0,1]
    confidence = max(0.0, min(1.0, float(confidence)))

    output = {
        "decision": decision,
        "confidence": round(confidence, 2),
        "requires_human": (decision == "review") or (len(low_conf) > 0),
        "reasons": reasons,
        "evidence": evidence,
        "flags": flags,
    }
    return output


# ---------- public entry ----------
def decide_claim(fields: Dict[str, Any], validation: Optional[Dict[str, Any]] = None,
                 summary: Optional[Dict[str, Any]] = None, full_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Public API: fields is expected to be the normalized fields dict used elsewhere.
    validation, summary, full_text are optional inputs that help the decision logic.
    """
    try:
        return rule_based_decision(fields=fields, validation=validation, summary=summary, full_text=full_text)
    except Exception:
        # defensive fallback: never raise, keep pipeline stable
        return {
            "decision": "review",
            "confidence": 0.4,
            "requires_human": True,
            "reasons": ["decision_engine_error"],
            "evidence": {"amount": None, "insured_amount": None, "low_confidence_fields": []},
            "flags": ["engine_error"]
        }

