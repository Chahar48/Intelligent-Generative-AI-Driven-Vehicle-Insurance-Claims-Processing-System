#  claim_pipeline/processing/decision_engine.py
"""
Tiny, crash-proof decision engine for the simplified claims pipeline.

Input:
- fields: dict of normalized fields, each like:
    { "original": ..., "value": "...", "success": True/False, "notes": "..." }

Output (always):
{
    "decision": "approve" | "review" | "reject",
    "confidence": float (0.0 - 1.0),
    "requires_human": bool,
    "reasons": [ "...", ... ],
    "evidence": {
        "amount": float|None,
        "insured_amount": float|None,
        "low_confidence_fields": [ ... ]
    }
}
"""

from typing import Dict, Any, List, Optional


# ---------- helpers ----------
def _get_value(field_name: str, fields: Dict[str, Any]) -> Optional[str]:
    f = fields.get(field_name)
    if isinstance(f, dict):
        return f.get("value") or ""
    return f or ""


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def _low_confidence_fields(fields: Dict[str, Any]) -> List[str]:
    """
    Here 'low confidence' = normalizer reported success=False
    but there was some original text (i.e. it was attempted but failed).
    """
    low = []
    for name, f in fields.items():
        if isinstance(f, dict):
            success = f.get("success", False)
            orig = f.get("original", "")
            # if normalization failed but original had content -> low confidence
            if not success and orig not in (None, "", []):
                low.append(name)
    return low


# ---------- core rule-based decision ----------
def rule_based_decision(fields: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    flags: List[str] = []

    # extract amounts
    amt_raw = _get_value("amount_estimated", fields)
    insured_raw = _get_value("insured_amount", fields)

    amt = _to_float(amt_raw)
    insured = _to_float(insured_raw)

    # basic checks
    if amt is None:
        reasons.append("amount_missing_or_invalid")
        flags.append("missing_amount")
    if insured is None:
        reasons.append("insured_amount_missing_or_invalid")
        flags.append("missing_insured_amount")

    # damage severity check (simple keyword scan)
    dmg = _get_value("damage", fields).lower() if _get_value("damage", fields) else ""
    high_risk_keywords = ["fire", "arson", "theft", "total loss", "totaled", "collision"]
    if any(k in dmg for k in high_risk_keywords):
        reasons.append("high_risk_damage_detected")
        flags.append("high_risk")

    # primary decision logic
    decision = "review"
    confidence = 0.6

    # if both amounts available, compare
    if amt is not None and insured is not None:
        if amt <= 0:
            decision = "reject"
            confidence = 0.95
            reasons.append("non_positive_claim_amount")
            flags.append("invalid_amount")
        elif amt > insured:
            decision = "reject"
            confidence = 0.95
            reasons.append("claimed_amount_exceeds_insured")
            flags.append("over_limit")
        else:
            # within insured amount
            ratio = amt / insured if insured > 0 else 0.0
            if ratio >= 0.9:
                decision = "review"
                confidence = 0.75
                reasons.append("amount_close_to_insured_limit")
                flags.append("near_limit")
            else:
                decision = "approve"
                confidence = 0.9
                reasons.append("amount_within_insured_limit")
                flags.append("within_limit")
    else:
        # missing amount info -> ask human
        decision = "review"
        confidence = 0.45
        reasons.append("insufficient_amount_information")

    # penalize confidence if many low-confidence fields
    low_conf = _low_confidence_fields(fields)
    if low_conf:
        confidence = max(0.0, confidence - 0.2)
        reasons.append("low_confidence_fields_present")
        flags.append("low_confidence")

    # additional small penalty if high-risk flagged
    if "high_risk" in flags:
        confidence = min(1.0, confidence + 0.0)  # keep same or adjust if you want

    # ensure confidence bounds
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    return {
        "decision": decision,
        "confidence": round(confidence, 2),
        "requires_human": decision == "review" or bool(low_conf),
        "reasons": reasons,
        "evidence": {
            "amount": amt,
            "insured_amount": insured,
            "low_confidence_fields": low_conf
        },
        "flags": flags
    }


# ---------- public entry ----------
def decide_claim(fields: Dict[str, Any], validation: Optional[Dict[str, Any]] = None, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    fields: normalized fields (dict)
    validation, summary: optional, not used in this simple version
    """
    return rule_based_decision(fields)
