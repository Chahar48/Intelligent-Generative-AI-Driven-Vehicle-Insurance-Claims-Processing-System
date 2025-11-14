# claim_pipeline/processing/field_extractor.py
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------
# REGEX PATTERNS
# -----------------------
RE_CLAIM_ID = re.compile(r"(?:Claim\s*(?:ID|No|#)|CLM)[:\s\-]*([A-Z0-9\-]+)", re.IGNORECASE)
RE_POLICY_NO = re.compile(r"(?:Policy\s*(?:No|Number|#)|PN)[:\s\-]*([A-Z0-9\-]+)", re.IGNORECASE)

RE_EMAIL = re.compile(r"\b([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[A-Za-z0-9.\-]+)\b")

# preserves +91 — allows digits, space, dash, parentheses
RE_PHONE = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")

RE_DATE = re.compile(
    r"(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
)

RE_NAME_LINE = re.compile(r"(?:Claimant|Insured|Name)[:\s\-]*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)")

RE_VEHICLE = re.compile(r"(?:Vehicle|Car|Model|Make|Registration)[:\s\-]*(.+)", re.IGNORECASE)

RE_DAMAGE = re.compile(r"(?:Damage|Loss|Damage\s*Description)[:\s\-]*(.+)", re.IGNORECASE)

# amount rules
RE_STRICT_AMOUNT = re.compile(r"(₹\s?\d[\d,]*|\$ ?\d[\d,]*|Rs\.? ?\d[\d,]*)", re.IGNORECASE)
RE_ESTIMATE = re.compile(
    r"(?:Estimated\s*(?:Amount|Cost)|Total\s*Amount|Amount\s*(?:Claimed|Estimated))"
    r"[:\s\-]*((?:₹|\$|Rs\.?)\s?\d[\d,]*)",
    re.IGNORECASE
)

# -----------------------
# HELPERS
# -----------------------
def _wrap(value, method="none", score=0.5):
    return {
        "value": value,
        "confidence": "high" if score >= 0.8 else "medium" if score >= 0.5 else "low",
        "score": score,
        "method": method,
    }

def _follow_multiline(lines: List[str], start: int) -> str:
    """Append next lines until a header or blank line appears."""
    collected = []
    for i in range(start + 1, min(start + 4, len(lines))):
        nxt = lines[i].rstrip()
        if not nxt:
            break
        if re.match(r"^[A-Za-z].{0,20}:\s*", nxt):  # header → stop
            break
        collected.append(nxt)
    return " ".join(collected)

# -----------------------
# MAIN FUNCTION
# -----------------------
def extract_claim_fields(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {
            "fields": {},
            "raw_lines": [],
            "notes": ["EMPTY_TEXT"]
        }

    lines = text.splitlines()
    notes = []

    fields = {
        "claim_id": _wrap(None),
        "policy_no": _wrap(None),
        "claimant_name": _wrap(None),
        "phone": _wrap(None),
        "email": _wrap(None),
        "vehicle": _wrap(None),
        "damage": _wrap(None),
        "amount_estimated": _wrap(None),
        "date": _wrap(None),
    }

    # --------------------------
    # LINE-BY-LINE EXTRACTION
    # --------------------------
    for idx, raw in enumerate(lines):

        # Claim ID
        m = RE_CLAIM_ID.search(raw)
        if m and not fields["claim_id"]["value"]:
            fields["claim_id"] = _wrap(m.group(1), "regex", 0.9)

        # Policy No
        m = RE_POLICY_NO.search(raw)
        if m and not fields["policy_no"]["value"]:
            fields["policy_no"] = _wrap(m.group(1), "regex", 0.9)

        # Name
        m = RE_NAME_LINE.search(raw)
        if m and not fields["claimant_name"]["value"]:
            fields["claimant_name"] = _wrap(m.group(1), "regex", 0.85)

        # Email
        m = RE_EMAIL.search(raw)
        if m and not fields["email"]["value"]:
            fields["email"] = _wrap(m.group(1), "regex", 0.9)

        # Phone — keep +91
        m = RE_PHONE.search(raw)
        if m and not fields["phone"]["value"]:
            fields["phone"] = _wrap(m.group(1), "regex", 0.9)

        # Date
        m = RE_DATE.search(raw)
        if m and not fields["date"]["value"]:
            fields["date"] = _wrap(m.group(1), "regex", 0.9)

        # Vehicle multiline
        m = RE_VEHICLE.search(raw)
        if m and not fields["vehicle"]["value"]:
            base = m.group(1).strip()
            extra = _follow_multiline(lines, idx)
            fields["vehicle"] = _wrap(f"{base} {extra}".strip(), "regex-multiline", 0.8)

        # Damage multiline
        m = RE_DAMAGE.search(raw)
        if m and not fields["damage"]["value"]:
            base = m.group(1).strip()
            extra = _follow_multiline(lines, idx)
            fields["damage"] = _wrap(f"{base} {extra}".strip(), "regex-multiline", 0.8)

    # --------------------------
    # AMOUNT EXTRACTION
    # --------------------------
    m = RE_ESTIMATE.search(text)
    if m:
        fields["amount_estimated"] = _wrap(m.group(1).strip(), "estimate-regex", 0.9)
    else:
        all_money = RE_STRICT_AMOUNT.findall(text)
        if all_money:
            fields["amount_estimated"] = _wrap(max(all_money, key=len), "amount-heuristic", 0.7)
        else:
            notes.append("missing:amount_estimated")

    # --------------------------
    # MISSING FIELD NOTES
    # --------------------------
    for k, v in fields.items():
        if not v["value"]:
            notes.append(f"missing:{k}")

    return {
        "fields": fields,
        "raw_lines": [{"index": i, "text": ln} for i, ln in enumerate(lines)],
        "notes": notes
    }
