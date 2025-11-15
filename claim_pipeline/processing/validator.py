# claim_pipeline/processing/validator.py
"""
Improved, robust validator for claim fields used by the claims pipeline.

Input: raw_fields (mapping field_name -> primitive value or None)
This function will call the project's normalizers and return a structured validation dict.

Output keys (kept compatible with pipeline expectations):
{
    "fields": { <field>: <normalized_dict>, ... },
    "issues": [...],
    "warnings": [...],
    "missing_required": [...],
    "risk_score": float(0..1),
    "recommendation": "approve"|"review"|"reject",
    "decision_reasons": [...],
    # additional optional keys for explainability:
    "is_complete": bool,
    "explainability": { ... }
}
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re

from claim_pipeline.processing.normalizer import (
    normalize_amount,
    normalize_date,
    normalize_phone,
    normalize_email,
    normalize_string,
)

# Minimal required fields for a claim to be processable
REQUIRED = ["claim_id", "policy_no", "date", "amount_estimated"]

# thresholds
LOW_CONF_SCORE = 0.6  # if normalizer had a numeric score lower than this consider low confidence


def _as_norm_dict(norm_candidate: Any) -> Dict[str, Any]:
    """
    Normalizer outputs may be:
     - a dict like {"original","value","success","notes", ...}
     - or a primitive value (string/None)
    Normalize to a predictable dict shape for downstream logic.
    """
    if isinstance(norm_candidate, dict):
        return {
            "original": norm_candidate.get("original", norm_candidate.get("value")),
            "value": norm_candidate.get("value", None),
            "success": bool(norm_candidate.get("success")) if "success" in norm_candidate else (norm_candidate.get("value") not in (None, "")),
            "notes": norm_candidate.get("notes") if isinstance(norm_candidate.get("notes"), (list, str)) else [],
            "confidence": norm_candidate.get("confidence", None),
            "score": norm_candidate.get("score", None),
            "raw_candidates": norm_candidate.get("raw_candidates", [])  # optional richer normalizers
        }
    else:
        return {
            "original": norm_candidate,
            "value": norm_candidate if norm_candidate not in (None, "") else None,
            "success": False if norm_candidate in (None, "") else True,
            "notes": [],
            "confidence": None,
            "score": None,
            "raw_candidates": []
        }


def _is_low_confidence_field(norm: Dict[str, Any]) -> bool:
    """Detect fields that were attempted but low confidence."""
    # attempted but explicitly failed
    if norm.get("success") is False and norm.get("original") not in (None, "", []):
        return True
    # numeric score present
    sc = norm.get("score")
    if isinstance(sc, (int, float)) and sc < LOW_CONF_SCORE:
        return True
    # textual confidence label
    conf = (norm.get("confidence") or "").lower() if norm.get("confidence") else None
    if conf in ("low", "very low", "weak"):
        return True
    return False


def _parse_float_safe(s: Optional[Any]) -> Optional[float]:
    """Try to parse float from str/int/float; return None on failure."""
    if s is None:
        return None
    try:
        if isinstance(s, (int, float)):
            return float(s)
        st = str(s).strip()
        st_clean = re.sub(r"[^\d\.\-]", "", st)
        if st_clean in ("", "-", "."):
            return None
        return float(st_clean)
    except Exception:
        return None


def _suspicious_email_domain(email_value: str) -> bool:
    """Simple heuristics for suspicious domains (typo-like endings)."""
    try:
        domain = email_value.split("@", 1)[1].lower()
    except Exception:
        return False
    suspicious_endings = (".con", ".cm", ".coo", ".gmal", ".hotnail")
    return any(domain.endswith(s) for s in suspicious_endings)


def _suspicious_phone_digits(digits: str) -> bool:
    """Detect placeholder or bogus phone patterns."""
    if not digits:
        return False
    # all same digit or sequential patterns
    if len(set(digits)) == 1:
        return True
    if digits.endswith("0000") or digits.startswith("000"):
        return True
    return False


def validate_claim_fields(raw_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main validator entry.

    raw_fields: mapping field_name -> primitive (as produced by extractor).
    Returns structured validation dict (see module docstring).
    """
    try:
        # -------------------------
        # 1) Normalize all fields (use existing normalizers)
        # -------------------------
        fields: Dict[str, Dict[str, Any]] = {}

        fields["claim_id"] = _as_norm_dict(normalize_string(raw_fields.get("claim_id")))
        fields["policy_no"] = _as_norm_dict(normalize_string(raw_fields.get("policy_no")))
        fields["claimant_name"] = _as_norm_dict(normalize_string(raw_fields.get("claimant_name")))
        fields["phone"] = _as_norm_dict(normalize_phone(raw_fields.get("phone")))
        fields["email"] = _as_norm_dict(normalize_email(raw_fields.get("email")))
        fields["vehicle"] = _as_norm_dict(normalize_string(raw_fields.get("vehicle")))
        fields["damage"] = _as_norm_dict(normalize_string(raw_fields.get("damage")))
        fields["date"] = _as_norm_dict(normalize_date(raw_fields.get("date")))
        fields["amount_estimated"] = _as_norm_dict(normalize_amount(raw_fields.get("amount_estimated")))

        # insured_amount may be missing
        if raw_fields.get("insured_amount") not in (None, ""):
            fields["insured_amount"] = _as_norm_dict(normalize_amount(raw_fields.get("insured_amount")))
        else:
            fields["insured_amount"] = {
                "original": None,
                "value": None,
                "success": False,
                "notes": ["not_provided"],
                "confidence": None,
                "score": None,
                "raw_candidates": []
            }

        # -------------------------
        # 2) Basic presence, missing & duplicate checks
        # -------------------------
        issues: List[str] = []
        warnings: List[str] = []
        missing_required: List[str] = []
        decision_reasons: List[str] = []

        for req in REQUIRED:
            nf = fields.get(req)
            if not nf or not nf.get("success") or nf.get("value") in (None, ""):
                missing_required.append(req)
                issues.append(f"{req}_missing")

        # duplicate detection if normalizer provided multiple candidates
        for fname in ("policy_no", "claim_id", "phone", "email"):
            nf = fields.get(fname, {})
            candidates = nf.get("raw_candidates") or []
            if isinstance(candidates, list) and len(candidates) > 1:
                warnings.append(f"{fname}_multiple_candidates")
                # add note to explainability
                # but keep processing

        # -------------------------
        # 3) Amount logic: parse numeric canonical amounts
        # -------------------------
        amt_val = _parse_float_safe(fields["amount_estimated"].get("value"))
        insured_val = _parse_float_safe(fields["insured_amount"].get("value"))

        # ignore obviously placeholder/office-use amounts (validator doesn't rely on raw_text here)
        if amt_val is None:
            # try fallback to 'original' (sometimes normalizer cleaned but left canonical blank)
            amt_val = _parse_float_safe(fields["amount_estimated"].get("original"))
        if insured_val is None:
            insured_val = _parse_float_safe(fields["insured_amount"].get("original"))

        if amt_val is None:
            issues.append("amount_missing_or_invalid")
        if insured_val is None:
            warnings.append("insured_amount_missing_or_invalid")

        # cross-field rule
        if amt_val is not None and insured_val is not None:
            if amt_val > insured_val:
                issues.append("amount_exceeds_insured")
                decision_reasons.append("Claimed amount greater than insured amount.")
            if amt_val <= 0:
                issues.append("non_positive_amount")
                decision_reasons.append("Claim amount is zero or negative.")

        # -------------------------
        # 4) Date plausibility
        # -------------------------
        date_ok = fields["date"].get("success") and fields["date"].get("value")
        date_str = fields["date"].get("value")
        if not date_ok:
            warnings.append("date_invalid_format")
        else:
            try:
                parsed = datetime.fromisoformat(date_str)
                # future check
                if parsed > (datetime.utcnow() + timedelta(days=1)):
                    warnings.append("date_in_future")
                # very old check (>5 years)
                if parsed < (datetime.utcnow() - timedelta(days=365 * 5)):
                    warnings.append("date_too_old")
            except Exception:
                warnings.append("date_validation_parse_failed")

        # -------------------------
        # 5) Phone / Email heuristics
        # -------------------------
        phone_val = fields["phone"].get("value") or ""
        phone_digits = re.sub(r"[^\d]", "", str(phone_val) or "")
        if fields["phone"].get("original") and not fields["phone"].get("success"):
            warnings.append("phone_normalization_failed")
        if phone_digits and _suspicious_phone_digits(phone_digits):
            warnings.append("phone_suspicious")

        email_val = (fields["email"].get("value") or "").lower()
        if fields["email"].get("original") and not fields["email"].get("success"):
            warnings.append("email_normalization_failed")
        if email_val and _suspicious_email_domain(email_val):
            warnings.append("email_domain_suspicious")

        # -------------------------
        # 6) Low-confidence fields detection (for AI/HITL)
        # -------------------------
        low_confidence_fields: List[str] = []
        for name, nf in fields.items():
            if _is_low_confidence_field(nf):
                low_confidence_fields.append(name)

        # -------------------------
        # 7) Risk scoring (explainable heuristic)
        # -------------------------
        risk = 0.10
        # penalties for missing required
        risk += 0.20 * len(missing_required)
        # penalties for concrete issues (higher weight)
        risk += 0.18 * len([i for i in issues if "amount" in i or "non_positive" in i])
        # penalties for warnings
        risk += 0.05 * len(warnings)
        # penalty for low confidence fields
        risk += 0.05 * len(low_confidence_fields)
        # extra heavy penalty if claimed > insured
        if "amount_exceeds_insured" in issues:
            risk += 0.35

        risk_score = max(0.0, min(1.0, risk))

        # -------------------------
        # 8) Recommendation logic
        # -------------------------
        if "amount_exceeds_insured" in issues:
            recommendation = "reject"
        elif missing_required:
            recommendation = "review"
        elif risk_score >= 0.7:
            recommendation = "review"
        else:
            recommendation = "approve"

        if not decision_reasons:
            if recommendation == "approve":
                decision_reasons.append("All required fields valid.")
            elif recommendation == "review":
                decision_reasons.append("Some fields need manual review.")
            elif recommendation == "reject":
                decision_reasons.append("Critical violation detected.")

        # -------------------------
        # 9) Explainability & final shape
        # -------------------------
        explainability = {
            "low_confidence_threshold": LOW_CONF_SCORE,
            "low_confidence_fields": sorted(low_confidence_fields),
            "field_success_map": {k: bool(v.get("success")) for k, v in fields.items()},
            "field_notes": {k: (v.get("notes") or []) for k, v in fields.items()},
        }

        result = {
            "fields": fields,
            "issues": sorted(list(set(issues))),
            "warnings": sorted(list(set(warnings))),
            "missing_required": sorted(list(set(missing_required))),
            "risk_score": round(risk_score, 3),
            "recommendation": recommendation,
            "decision_reasons": decision_reasons,
            # extras that are safe for pipeline components to read if present:
            "is_complete": len(missing_required) == 0,
            "explainability": explainability,
        }

        return result

    except Exception:
        # Defensive fallback to keep pipeline stable
        return {
            "fields": {},
            "issues": ["validator_error"],
            "warnings": [],
            "missing_required": REQUIRED.copy(),
            "risk_score": 1.0,
            "recommendation": "review",
            "decision_reasons": ["validator_exception_fallback"],
            "is_complete": False,
            "explainability": {}
        }

