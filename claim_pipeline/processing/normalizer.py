# claim_pipeline/processing/normalizer.py
"""
Improved intelligent normalizers.
- OCR-aware numeric fixes
- Strong amount extraction
- Robust date parsing with OCR cleanup
- Smart phone cleanup + heuristics
- Email autocorrection (domain fixes, (at) fixes)
- Unicode + ligature cleanup for strings

Output format ALWAYS:
{
    "original": raw,
    "value": cleaned_value_or_empty_string,
    "success": True/False,
    "notes": "short_reason"
}
"""

import re
import unicodedata
from datetime import datetime
from dateutil import parser as date_parser


# -------------------------
# Standard output wrapper
# -------------------------
def make_out(raw, value, success, notes=""):
    return {
        "original": raw,
        "value": value if value is not None else "",
        "success": success,
        "notes": notes
    }


# ==========================================================
# === OCR FIX UTILITIES ====================================
# ==========================================================

def fix_ocr_numbers(s: str) -> str:
    """Fix common OCR errors for numeric content."""
    if not s:
        return s

    s = str(s)

    # Common replacements: O→0, o→0, l→1, I→1
    s = re.sub(r'(?<=\d)[Oo](?=\d)', "0", s)
    s = re.sub(r'[Oo](?=\d)', "0", s)
    s = re.sub(r'(?<=\d)[Oo]', "0", s)

    s = re.sub(r'(?<=\d)[lI](?=\d)', "1", s)
    s = re.sub(r'[lI](?=\d)', "1", s)
    s = re.sub(r'(?<=\d)[lI]', "1", s)

    # Remove zero-width spaces
    s = s.replace("\u200b", "")

    # Remove stray characters
    s = re.sub(r"[^\d.,\-₹Rs$ ]", "", s)

    return s.strip()


def clean_unicode(s: str) -> str:
    """Normalize ligatures and fix OCR punctuation."""
    if not s:
        return s

    ligatures = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"'
    }

    for bad, good in ligatures.items():
        s = s.replace(bad, good)

    s = unicodedata.normalize("NFKC", s)
    return s


# ==========================================================
# === AMOUNT NORMALIZER ====================================
# ==========================================================

def normalize_amount(raw):
    if not raw:
        return make_out(raw, "", False, "empty_input")

    s = clean_unicode(str(raw)).strip()
    s = fix_ocr_numbers(s)

    # Remove currency labels
    s = re.sub(r"(Rs\.?|₹|INR|USD|\$)", "", s, flags=re.IGNORECASE)
    s = s.replace(",", "").replace(" ", "")

    # Extract number (float allowed)
    match = re.search(r"-?\d+(\.\d+)?", s)
    if not match:
        return make_out(raw, "", False, "no_numeric_found")

    try:
        value = float(match.group(0))
    except:
        return make_out(raw, "", False, "parse_failed")

    # Sanity check (remove impossible values)
    if abs(value) > 1e10:
        return make_out(raw, "", False, "unrealistic_amount")

    return make_out(raw, f"{value:.2f}", True, "normalized")


# ==========================================================
# === DATE NORMALIZER =======================================
# ==========================================================

def normalize_date(raw):
    if not raw:
        return make_out(raw, "", False, "empty_input")

    s = clean_unicode(str(raw)).strip()

    # OCR fixes: O→0, I→1 inside date contexts
    s = fix_ocr_numbers(s)

    # Remove ordinal suffixes (12th → 12)
    s = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", s, flags=re.IGNORECASE)

    # If already ISO
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return make_out(raw, s, True, "iso_format")

    try:
        dt = date_parser.parse(s, fuzzy=True)
        # Basic sanity year check
        if dt.year < 1900 or dt.year > datetime.utcnow().year + 1:
            return make_out(raw, "", False, "year_out_of_range")

        return make_out(raw, dt.date().isoformat(), True, "parsed")
    except:
        return make_out(raw, "", False, "parse_failed")


# ==========================================================
# === PHONE NORMALIZER =====================================
# ==========================================================

def normalize_phone(raw):
    if not raw:
        return make_out(raw, "", False, "empty_input")

    s = clean_unicode(str(raw))
    digits = re.sub(r"[^\d]", "", s)

    if not (7 <= len(digits) <= 15):
        return make_out(raw, "", False, "invalid_length")

    # Indian mobile (best case)
    if len(digits) == 10 and digits[0] in "6789":
        return make_out(raw, "+91" + digits, True, "indian_format")

    # Leading zero case
    if len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        return make_out(raw, "+91" + digits[1:], True, "leading_zero_indian")

    # Generic international format
    return make_out(raw, "+" + digits, True, "generic_format")


# ==========================================================
# === EMAIL NORMALIZER =====================================
# ==========================================================

COMMON_DOMAINS = {
    "gmai.com": "gmail.com",
    "gmial.com": "gmail.com",
    "hotnail.com": "hotmail.com",
    "outllok.com": "outlook.com",
    "yaho.com": "yahoo.com",
    "icloud.con": "icloud.com"
}

def normalize_email(raw):
    if not raw:
        return make_out(raw, "", False, "empty_input")

    s = clean_unicode(str(raw)).strip().lower().replace(" ", "")

    # fix "(at)" → "@"
    s = s.replace("(at)", "@")

    # basic missing @ fix
    if "@" not in s:
        # try converting "name at gmail.com"
        s = re.sub(r"\bat\b", "@", s)
        if "@" not in s:
            return make_out(raw, "", False, "missing_at_symbol")

    if "@" not in s:
        return make_out(raw, "", False, "invalid_email")

    local, domain = s.split("@", 1)

    # autocorrect common domain mistakes
    for bad, good in COMMON_DOMAINS.items():
        if bad in domain:
            domain = good

    candidate = f"{local}@{domain}"

    if not re.match(r"^[a-z0-9_.+\-]+@[a-z0-9\-]+\.[a-z]{2,}$", candidate):
        return make_out(raw, "", False, "invalid_email_format")

    return make_out(raw, candidate, True, "valid")


# ==========================================================
# === STRING NORMALIZER ====================================
# ==========================================================

def normalize_string(raw):
    if not raw:
        return make_out(raw, "", False, "empty_input")

    s = clean_unicode(str(raw))
    s = re.sub(r"\s+", " ", s).strip()

    return make_out(raw, s, True, "cleaned")

