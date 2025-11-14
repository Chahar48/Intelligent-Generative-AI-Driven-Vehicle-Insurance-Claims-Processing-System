# claim_pipeline/processing/validator.py
"""
Advanced validator for claims processing pipeline.

Expects the normalizer functions to return structured dicts with keys such as:
{
  "original": ...,
  "value": ...,
  "canonical": ...,
  "confidence": "low"|"medium"|"high",
  "score": 0.0-1.0,
  "corrected": bool,
  "notes": [...],
  "metadata": {...}
}

Produces a structured validation result:
{
  "fields": { <field>: <normalized_dict>, ... },
  "issues": [...],
  "warnings": [...],
  "notes": [...],
  "missing_required": [...],
  "ai_fields": [...],            # fields suggested for AI inference
  "risk_score": 0.0-1.0,
  "is_complete": bool,
  "recommendation": "approve"|"review"|"reject"|"escalate",
  "decision_reasons": [...],     # short explanation pieces
  "explainability": { ... }      # metadata useful for audit/UI
}
"""

import logging
from typing import Dict, Any, List, Optional

# Import the structured normalizers (adjust package path if different)
from claim_pipeline.processing.normalizer import (
    normalize_amount,
    normalize_date,
    normalize_phone,
    normalize_email,
    normalize_string,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ----------------------
# Configuration
# ----------------------
DEFAULT_REQUIRED_FIELDS = ["claim_id", "policy_no", "amount_estimated", "date"]
LOW_CONFIDENCE_THRESHOLD = 0.5  # score below this is considered low confidence


# ----------------------
# Helper functions
# ----------------------
def _is_missing_or_empty(norm_field: Dict[str, Any]) -> bool:
    return (not norm_field) or (norm_field.get("value") is None)


def _is_low_confidence(norm_field: Dict[str, Any], thresh: float = LOW_CONFIDENCE_THRESHOLD) -> bool:
    # Some normalizers may omit 'score' — treat missing score conservatively
    score = norm_field.get("score") if isinstance(norm_field, dict) else None
    if score is None:
        # fallback to textual label if present
        if norm_field.get("confidence") in ("low", None):
            return True
        return False
    return float(score) < float(thresh)


def _field_label(name: str) -> str:
    # friendly labels for UI/logs
    return name.replace("_", " ").title()


# ----------------------
# Main validation function
# ----------------------
def validate_claim_fields(raw_fields: Dict[str, Any], required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate and return structured info including normalized values, issues, warnings,
    risk score, AI-needed field list, and a recommendation.
    """
    if required_fields is None:
        required_fields = DEFAULT_REQUIRED_FIELDS

    issues: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    decision_reasons: List[str] = []

    # ----------------------
    # Normalize all fields (using structured normalizers)
    # ----------------------
    # Each normalizer returns a structured dict as described in module docstring
    fields: Dict[str, Dict[str, Any]] = {}

    fields["claim_id"] = normalize_string(raw_fields.get("claim_id"))
    fields["policy_no"] = normalize_string(raw_fields.get("policy_no"))
    fields["claimant_name"] = normalize_string(raw_fields.get("claimant_name"))
    fields["phone"] = normalize_phone(raw_fields.get("phone"))
    fields["email"] = normalize_email(raw_fields.get("email"))
    fields["vehicle"] = normalize_string(raw_fields.get("vehicle"))
    fields["damage"] = normalize_string(raw_fields.get("damage"))
    fields["date"] = normalize_date(raw_fields.get("date"))
    fields["amount_estimated"] = normalize_amount(raw_fields.get("amount_estimated"))
    fields["insured_amount"] = normalize_amount(raw_fields.get("insured_amount")) if raw_fields.get("insured_amount") else {
        "original": None, "value": None, "canonical": None,
        "confidence": "low", "score": 0.0, "corrected": False,
        "notes": ["not_provided"], "metadata": {}
    }

    # ----------------------
    # Basic presence & confidence checks
    # ----------------------
    ai_fields: List[str] = []  # fields recommended for AI inference/fallback

    for fname, norm in fields.items():
        # Missing critical field
        if _is_missing_or_empty(norm):
            issues.append(f"{fname}:missing")
            logger.debug(f"[validator] Field missing: {fname}")
        # Low confidence
        if isinstance(norm, dict) and _is_low_confidence(norm):
            warnings.append(f"{fname}:low_confidence")
            # if missing or low confidence, request AI fallback
            if norm.get("value") is None or norm.get("score", 0.0) < LOW_CONFIDENCE_THRESHOLD:
                ai_fields.append(fname)
        # If corrected by normalizer, surface as a warning
        if isinstance(norm, dict) and norm.get("corrected"):
            warnings.append(f"{fname}:auto_corrected")
        # Keep notes from normalizer
        if isinstance(norm, dict) and norm.get("notes"):
            notes.extend([f"{fname}:{n}" for n in norm.get("notes")])

    # ----------------------
    # Business logic checks & cross-field validations
    # ----------------------
    # Amount consistency with insured_amount
    amt = fields["amount_estimated"].get("value")
    insured = fields["insured_amount"].get("value")
    if amt is not None and insured is not None:
        if amt > insured:
            issues.append("amount_estimated_exceeds_insured_amount")
            decision_reasons.append("Claimed amount exceeds insured amount.")
            logger.info("[validator] Claimed amount exceeds insured amount.")

    # Check zero or negative claim amount — suspicious/invalid
    if amt is not None and amt <= 0:
        issues.append("amount_nonpositive")
        decision_reasons.append("Claimed amount is zero or negative.")
        logger.info("[validator] Non-positive claimed amount detected.")

    # Date checks: ensure parsed date is not in distant past/future
    date_norm = fields["date"]
    if date_norm and date_norm.get("value"):
        try:
            from datetime import datetime, timedelta
            parsed_iso = date_norm["value"]
            parsed_dt = datetime.fromisoformat(parsed_iso)
            # if claim date > now + 1 day -> future date suspicious
            if parsed_dt > (datetime.utcnow() + timedelta(days=1)):
                warnings.append("date_in_future")
                notes.append("Claim date is in the future.")
            # if claim date older than, say, 5 years -> warn
            if parsed_dt < (datetime.utcnow() - timedelta(days=365 * 5)):
                warnings.append("date_old")
                notes.append("Claim date is older than 5 years.")
        except Exception:
            warnings.append("date_validation_failed")

    # Email/phone plausibility checks
    email_norm = fields["email"]
    if email_norm and email_norm.get("value"):
        # suspicious domain heuristics can be added here (typosquatting)
        domain = email_norm["value"].split("@")[-1]
        if domain.endswith(".con") or domain.endswith(".cm"):
            warnings.append("email_domain_suspicious")
            notes.append(f"Suspicious email domain: {domain}")

    phone_norm = fields["phone"]
    if phone_norm and phone_norm.get("value"):
        digits = phone_norm["value"]
        # very basic invalid phone patterns
        if set(digits) == {"0"} or set(digits) == {"1"}:
            warnings.append("phone_suspicious")
            notes.append("Phone number looks like a placeholder.")

    # Duplicate/conflict detection (e.g., multiple policy numbers or claim ids in raw text)
    # If field has raw_candidates in normalizer output, surface duplicates
    for fname in ["policy_no", "claim_id", "phone", "email"]:
        norm = fields.get(fname, {})
        if isinstance(norm, dict):
            candidates = norm.get("raw_candidates") or []
            if len(candidates) > 1:
                warnings.append(f"{fname}:multiple_candidates")
                notes.append(f"{fname} found multiple candidates: {candidates}")

    # ----------------------
    # Completeness and required fields
    # ----------------------
    missing_required: List[str] = []
    for r in required_fields:
        # required check uses normalized .value presence
        nf = fields.get(r)
        if nf is None or nf.get("value") is None:
            missing_required.append(r)

    is_complete = len(missing_required) == 0

    # ----------------------
    # Risk scoring (simple heuristic aggregator)
    # ----------------------
    # Baseline
    risk = 0.10
    # Penalty for missing required fields
    risk += 0.20 * len(missing_required)
    # Penalty for issues
    risk += 0.15 * sum(1 for i in issues)
    # Penalty for low-confidence fields
    low_conf_count = sum(1 for f in fields.values() if isinstance(f, dict) and _is_low_confidence(f))
    risk += 0.05 * low_conf_count
    # Penalty for amount/insured mismatch
    if "amount_estimated_exceeds_insured_amount" in issues:
        risk += 0.40
    # Cap
    risk_score = max(0.0, min(1.0, risk))

    # ----------------------
    # Decision logic (explainable)
    # ----------------------
    # Priority rules (reasons collected above)
    recommendation = "approve"
    explain_reasons: List[str] = []

    # If critical issues exist, force reject
    if any(k.startswith("amount_estimated_exceeds_insured_amount") or "amount_estimated_exceeds_insured_amount" in k for k in issues):
        recommendation = "reject"
        explain_reasons.append("Critical: claimed amount > insured amount.")
    # If required fields missing -> review
    elif missing_required:
        recommendation = "review"
        explain_reasons.append(f"Missing required fields: {', '.join(missing_required)}")
    # Escalate for very high risk
    elif risk_score >= 0.8:
        recommendation = "escalate"
        explain_reasons.append("Very high risk score.")
    elif risk_score >= 0.6:
        recommendation = "review"
        explain_reasons.append("High risk score.")
    else:
        recommendation = "approve"
        explain_reasons.append("Low risk; auto-approve.")

    # Add warnings/issues reasons to explain_reasons
    if warnings:
        explain_reasons.append("Warnings present: " + "; ".join(warnings))
    if issues:
        explain_reasons.append("Issues present: " + "; ".join(issues))

    # ----------------------
    # Final structured response
    # ----------------------
    result: Dict[str, Any] = {
        "fields": fields,
        "issues": issues,
        "warnings": warnings,
        "notes": notes,
        "missing_required": missing_required,
        "ai_fields": sorted(list(set(ai_fields))),  # unique
        "risk_score": round(risk_score, 3),
        "is_complete": is_complete,
        "recommendation": recommendation,
        "decision_reasons": explain_reasons,
        "explainability": {
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "required_fields": required_fields,
            "field_scores": {k: (v.get("score") if isinstance(v, dict) else None) for k, v in fields.items()}
        }
    }

    logger.info(f"[validator] Recommendation: {recommendation} (risk={result['risk_score']})")
    return result
