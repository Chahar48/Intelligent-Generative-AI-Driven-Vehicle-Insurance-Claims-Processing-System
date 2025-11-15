###############################################New Code for Better Extraction techniques####################################################################
"""
Robust, dependency-free field extractor for the claims pipeline.

Returns a dict with shape:
{
  "fields": {
    "claim_id": {"value":"CL/MV/2024/78542", "canonical":"CL/MV/2024/78542", "score":0.95, "confidence":"high", "method":"regex"},
    ...
  },
  "raw_lines": [{"index":0, "text": "..."}, ...],
  "notes": ["..."]
}

Key features:
- Strong pattern matching for real-world claim forms
- Supports OCR text, scanned text, noisy text
- Extracts: claim_id, policy_no, claimant_name, phone, email,
            vehicle, registration_no, make, model, year,
            damage, amount_estimated, insured_amount, date
- High defensive stability: never crashes, pipeline-safe
"""

from __future__ import annotations
import re
from typing import Dict, Any, List, Optional

# =============================================================
# REGEX PATTERNS
# =============================================================

# --- Claim ID ---
RE_CLAIM_ID = re.compile(
    r"""(?ix)
    \b
    (?:claim\s*(?:reference|ref|id|no|number))   # MUST contain label
    [\s:\-]*
    ([A-Z0-9][A-Z0-9\/\-]{4,})                   # at least 5 chars
    \b
    """
)
# RE_CLAIM_ID = re.compile(
#     r"""(?ix)
#     \b
#     (?:
#         claim\s*(?:reference|ref|id|no|number)? |
#         cl(?:aim)?[.\s:_\-]*ref |
#         c[\s\._\-]*ref |
#         clm |
#         cl[/:\-] |
#         claimref |
#         claim-id |
#         claim-no
#     )
#     [\s:\-]*
#     (
#         [A-Z0-9]{2,}[A-Z0-9\/\-\_]{2,}
#     )
#     \b
#     """
# )

# --- Policy Number ---
RE_POLICY_NO = re.compile(
    r"""(?ix)
    \b
    (?:
        policy\s*(?:number|no|id)? |
        pol(?:icy)?\s*no |
        pol\s*id |
        policy[-_ ]?no |
        policynum
    )
    [\s:\-]*
    ([A-Z0-9][A-Z0-9\/\-\_]{3,})
    \b
    """
)

# --- Email ---
RE_EMAIL = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# --- Phone ---
RE_PHONE = re.compile(
    r"""(?ix)
    (
        \+?\d{1,3}
        [\s\-\(\)]*
        \d{2,5}
        (?:[\s\-]?\d){5,12}
    )
    """
)

# --- Date formats ---
RE_DATE = re.compile(
    r"""(?ix)
    \b(
        \d{4}[-/]\d{2}[-/]\d{2} |
        \d{1,2}[-/]\d{1,2}[-/]\d{4} |
        \d{1,2}\s+
        (?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|
         january|february|march|april|may|june|july|august|
         september|october|november|december)
        \s+\d{4}
    )\b
    """
)

# --- Name extraction ---
RE_NAME_LINE = re.compile(
    r"""(?ix)
    (?:full\s*name|claimant\s*name|insured\s*name)
    [\s:\-]*
    ([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,3})
    """
)
# RE_NAME_LINE = re.compile(
#     r"""(?ix)
#     (?:
#         full\s*name|
#         claimant\s*name|
#         insured\s*name|
#         claimant|
#         insured|
#         customer\s*name|
#         client\s*name|
#         name
#     )
#     [\s:\-]*
#     ([A-Z][A-Za-z'`\.\-]+(?:\s+[A-Z][A-Za-z'`\.\-]+){0,5})
#     """
# )

# --- Vehicle Registration ---
RE_REGISTRATION = re.compile(
    r"""(?ix)
    \b
    (?:[A-Z]{2,3}[\s\-]?\d{1,4}[\s\-]?[A-Z]{1,4}[\s\-]?\d{1,4})
    \b
    """
)

# --- Damage Description ---
RE_DAMAGE = re.compile(
    r"""(?ix)
    (?:
        nature\s*of\s*damage |
        damage\s*description |
        nature\s*of\s*loss |
        accident\s*damage |
        damage |
        loss
    )
    [\s:\-]*
    (.+)
    """
)

# --- Amounts ---
RE_AMOUNT = re.compile(
    r"""(?ix)
    (₹|rs\.?|inr|\$)?\s*
    (
        \d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})? |
        \d+(?:\.\d{1,2})?
    )
    """
)

RE_ESTIMATE_LABEL = re.compile(
    r"""(?ix)
    (?:estimated\s*(?:repair\s*)?(?:cost|amount|loss|claim|value)|
       total\s*estimated\s*(?:cost|amount|loss)|
       repair\s*estimate)
    [\s:\-]*
    """
)

# =============================================================
# FALLBACK FIELD KEYWORD MAP
# =============================================================

FIELD_KEYWORDS = {
    "claim_id": ["claim reference", "claim ref", "claim id", "claim number", "clm", "claim no"],
    "policy_no": ["policy number", "policy no", "policy id", "pn"],
}

# =============================================================
# HEADER DETECTION FOR MULTILINE CAPTURE
# =============================================================

RE_HEADER_LINE = re.compile(
    r"""(?ix)
    ^
    (?:
        [A-Za-z0-9][A-Za-z0-9\s\-\_/\.]{0,40}[:\-]\s*$ |
        [A-Z][A-Z\s]{4,50}$
    )
    """
)

# =============================================================
# HELPERS
# =============================================================

def _empty_field(method: str = "none") -> Dict[str, Any]:
    return {"value": None, "canonical": None, "score": 0.0, "confidence": "low", "method": method}

def _wrap(value: Any, score: float = 0.6, method: str = "heuristic") -> Dict[str, Any]:
    conf = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
    return {"value": value, "canonical": value, "score": round(score, 3), "confidence": conf, "method": method}

def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def _is_gibberish(text: str) -> bool:
    printable = sum(1 for c in text if 32 <= ord(c) <= 126 or c in "\n\r\t")
    return printable / max(1, len(text)) < 0.55

def _digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

def _normalize_amount_match(m: re.Match) -> Optional[str]:
    if not m:
        return None
    cur = m.group(1) or ""
    num = m.group(2) or ""
    clean = re.sub(r"[\s,]", "", num)
    try:
        val = float(clean)
        return (cur + " " if cur else "") + f"{val:.2f}"
    except:
        return None

def _collect_multiline(lines: List[str], idx: int, max_lines: int = 3) -> str:
    out = []
    for i in range(idx + 1, min(len(lines), idx + 1 + max_lines)):
        ln = lines[i].strip()
        if not ln or RE_HEADER_LINE.match(ln):
            break
        out.append(ln)
    return " ".join(out)

# =============================================================
# MAIN EXTRACTOR
# =============================================================

def extract_claim_fields(text: str) -> Dict[str, Any]:
    try:
        if not text or not isinstance(text, str):
            return {"fields": {}, "raw_lines": [], "notes": ["EMPTY_TEXT"]}

        if _is_gibberish(text) and len(text) > 2000:
            return {"fields": {}, "raw_lines": [], "notes": ["BINARY_OR_GIBBERISH_TEXT"]}

        lines = [l.rstrip("\n") for l in text.splitlines()]
        raw_lines = [{"index": i, "text": lines[i]} for i in range(len(lines))]
        notes: List[str] = []
        full_text = text

        # ---------- Initialize fields ----------
        fields = {
            "claim_id": _empty_field("init"),
            "policy_no": _empty_field("init"),
            "claimant_name": _empty_field("init"),
            "phone": _empty_field("init"),
            "email": _empty_field("init"),
            "vehicle": _empty_field("init"),
            "registration_no": _empty_field("init"),
            "make": _empty_field("init"),
            "model": _empty_field("init"),
            "year": _empty_field("init"),
            "damage": _empty_field("init"),
            "amount_estimated": _empty_field("init"),
            "insured_amount": _empty_field("init"),
            "date": _empty_field("init"),
        }

        # =====================================================
        # 1) STRONG DOCUMENT-WIDE REGEX EXTRACTION
        # =====================================================

        if m := RE_CLAIM_ID.search(full_text):
            fields["claim_id"] = _wrap(_clean(m.group(1)), 0.95, "regex")

        if m := RE_POLICY_NO.search(full_text):
            fields["policy_no"] = _wrap(_clean(m.group(1)), 0.95, "regex")

        if m := RE_EMAIL.search(full_text):
            fields["email"] = _wrap(_clean(m.group(0)), 0.95, "regex")

        if m := RE_REGISTRATION.search(full_text):
            fields["registration_no"] = _wrap(_clean(m.group(0)), 0.9, "regex")

        # =====================================================
        # 2) AMOUNT EXTRACTION
        # =====================================================

        amount_found = False
        for idx, ln in enumerate(lines):
            if RE_ESTIMATE_LABEL.search(ln):
                if am := RE_AMOUNT.search(ln):
                    canon = _normalize_amount_match(am)
                    if canon:
                        fields["amount_estimated"] = _wrap(canon, 0.95, "labelled")
                        amount_found = True
                        break

                nxt = _collect_multiline(lines, idx, max_lines=2)
                if nxt and (am2 := RE_AMOUNT.search(nxt)):
                    canon = _normalize_amount_match(am2)
                    if canon:
                        fields["amount_estimated"] = _wrap(canon, 0.9, "labelled-next")
                        amount_found = True
                        break

        if not amount_found and (m := RE_AMOUNT.search(full_text)):
            canon = _normalize_amount_match(m)
            if canon:
                fields["amount_estimated"] = _wrap(canon, 0.7, "heuristic")

        if m := re.search(r"Insured\s*Amount[:\s\-]*(₹|Rs\.?|INR|\$)?\s*([0-9,\.\s]+)", full_text, re.IGNORECASE):
            try:
                num = float(re.sub(r"[,\s]", "", m.group(2)))
                fields["insured_amount"] = _wrap(f"{num:.2f}", 0.92, "labelled")
            except:
                pass

        # =====================================================
        # 3) LINE-BY-LINE EXTRACTION
        # =====================================================

        for idx, raw in enumerate(lines):
            ln = raw.strip()
            if not ln:
                continue
            lower = ln.lower()

            # ---- CLAIMANT NAME ----
            if not fields["claimant_name"]["value"]:
                if m := RE_NAME_LINE.search(ln):
                    fields["claimant_name"] = _wrap(_clean(m.group(1)), 0.95, "regex-line")

            # ---- PHONE ----
            if not fields["phone"]["value"]:
                if m := RE_PHONE.search(ln):
                    ph = m.group(1)
                    dg = _digits(ph)
                    canon = "+" + dg if ph.strip().startswith("+") else dg
                    fields["phone"] = _wrap(canon, 0.9, "regex")

            # ---- DATE ----
            if not fields["date"]["value"]:
                if m := RE_DATE.search(ln):
                    fields["date"] = _wrap(_clean(m.group(1)), 0.9, "regex")

            # ---- DAMAGE ----
            if not fields["damage"]["value"]:
                if m := RE_DAMAGE.search(ln):
                    base = _clean(m.group(1))
                    extra = _collect_multiline(lines, idx)
                    fields["damage"] = _wrap((base + " " + extra).strip(), 0.85, "multiline")

            # ---- Registration No ----
            if not fields["registration_no"]["value"]:
                if m := re.search(r"Registration\s*Number[:\s\-]*([A-Z0-9\-\s]+)", ln, re.IGNORECASE):
                    fields["registration_no"] = _wrap(_clean(m.group(1)), 0.9, "regex")

            # ---- Make ----
            if not fields["make"]["value"]:
                if m := re.search(r"Make[:\s\-]*([A-Za-z0-9 ]+)", ln, re.IGNORECASE):
                    fields["make"] = _wrap(_clean(m.group(1)), 0.9, "regex")

            # ---- Model ----
            if not fields["model"]["value"]:
                if m := re.search(r"Model[:\s\-]*([A-Za-z0-9 ]+)", ln, re.IGNORECASE):
                    fields["model"] = _wrap(_clean(m.group(1)), 0.9, "regex")

            # ---- Year ----
            if not fields["year"]["value"]:
                if m := re.search(r"Year[:\s\-]*([0-9]{4})", ln, re.IGNORECASE):
                    fields["year"] = _wrap(m.group(1), 0.9, "regex")

            # ---- Vehicle (broad) ----
            if not fields["vehicle"]["value"]:
                if m := re.search(r"(?:Vehicle|Vehicle\s*Details|Vehicle\s*Information)[:\s\-]*(.+)", ln, re.IGNORECASE):
                    base = m.group(1).strip()
                    extra = _collect_multiline(lines, idx)
                    fields["vehicle"] = _wrap((base + " " + extra).strip(), 0.85, "multiline")

            # ---- Email fallback ----
            if not fields["email"]["value"]:
                if m := RE_EMAIL.search(ln):
                    fields["email"] = _wrap(_clean(m.group(0)), 0.95, "regex")

            # ---- KEYWORD FALLBACK FOR CLAIM ID ----
            if not fields["claim_id"]["value"]:
                for kw in FIELD_KEYWORDS["claim_id"]:
                    if kw in lower:
                        if m := re.search(r"[A-Z0-9\/\-_]{4,}", ln):
                            fields["claim_id"] = _wrap(_clean(m.group(0)), 0.75, "keyword")
                            break

            # ---- KEYWORD FALLBACK FOR POLICY NO ----
            if not fields["policy_no"]["value"]:
                for kw in FIELD_KEYWORDS["policy_no"]:
                    if kw in lower:
                        if m := re.search(r"[A-Z0-9\/\-_]{4,}", ln):
                            fields["policy_no"] = _wrap(_clean(m.group(0)), 0.75, "keyword")
                            break

        # =====================================================
        # 4) HEURISTICS / POST PROCESSING
        # =====================================================

        # Claimant Signature heuristic
        if not fields["claimant_name"]["value"]:
            for ln in lines:
                if m := re.search(r"Claimant\s*Signature[:\s\-]*([A-Za-z ,.]{3,60})", ln, re.IGNORECASE):
                    fields["claimant_name"] = _wrap(_clean(m.group(1)), 0.85, "signature")
                    break

        # TitleCase fallback (name guess)
        if not fields["claimant_name"]["value"]:
            for ln in lines[:20]:
                if re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$", ln.strip()):
                    fields["claimant_name"] = _wrap(_clean(ln.strip()), 0.7, "titlecase")
                    break

        # Fix OCR phone/date confusion
        if fields["phone"]["value"]:
            digits = _digits(str(fields["phone"]["value"]))
            if len(digits) in (6, 8):
                # try alternative occurrence
                for m in RE_PHONE.finditer(full_text):
                    d2 = _digits(m.group(1))
                    if d2 != digits:
                        fields["phone"] = _wrap("+" + d2 if m.group(1).startswith("+") else d2, 0.85, "alt-phone")
                        break

        # Normalize canonical amount values
        for key in ("amount_estimated", "insured_amount"):
            val = fields[key]["value"]
            if isinstance(val, str) and val:
                if mm := re.search(r"([0-9][0-9,\.]*)", val):
                    try:
                        num = float(re.sub(r"[,\s]", "", mm.group(1)))
                        fields[key]["canonical"] = f"{num:.2f}"
                    except:
                        pass

        # =====================================================
        # 5) NOTES FOR MISSING IMPORTANT FIELDS
        # =====================================================

        for k in ["claim_id", "policy_no", "claimant_name", "date", "amount_estimated"]:
            if not fields[k]["value"]:
                notes.append(f"missing_{k}")

        return {"fields": fields, "raw_lines": raw_lines, "notes": notes}

    except Exception:
        return {"fields": {}, "raw_lines": [], "notes": ["EXTRACTOR_ERROR"]}


