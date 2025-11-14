import re
import unicodedata
import logging
from datetime import datetime
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AMOUNT_UPPER_BOUND = 1e10
MAX_STRING_LEN = 2000

LIGATURES_MAP = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}

COMMON_DOMAIN_CORRECTIONS = {
    "gmai.com": "gmail.com",
    "gmial.com": "gmail.com",
    "hotnail.com": "hotmail.com",
    "outllok.com": "outlook.com",
    "icloud.con": "icloud.com",
}


def _score_label(score):
    if score >= 0.85:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


# ----------------------------------------------------------
# AMOUNT NORMALIZER
# ----------------------------------------------------------
def normalize_amount(raw):
    out = {
        "original": raw,
        "value": None,
        "canonical": None,
        "confidence": "low",
        "score": 0.0,
        "corrected": False,
        "notes": [],
        "metadata": {},
    }

    if not raw:
        out["notes"].append("empty_input")
        return out

    s = str(raw).strip()

    # Remove Rs. → Rs
    s = re.sub(r"\b(Rs)\.\b", r"\1", s, flags=re.IGNORECASE)

    # OCR fix: R5 → Rs
    if s.startswith("R5"):
        s = "Rs" + s[2:]
        out["corrected"] = True

    # Remove spaces
    s = s.replace(" ", "")

    # Detect currency
    currency = None
    if re.search(r"(Rs|₹)", s, re.IGNORECASE):
        currency = "INR"
    if "$" in s:
        currency = "USD"
    out["metadata"]["currency_hint"] = currency

    # Remove currency
    s = re.sub(r"(Rs|₹|\$|USD)", "", s, flags=re.IGNORECASE)

    # OCR FIXES BEFORE PARSING
    s = s.replace("OO", "00")
    s = re.sub(r"(?<=\d)[Oo](?=\d)", "0", s)
    s = re.sub(r"(?<=\d)[Oo]", "0", s)
    s = re.sub(r"[Oo](?=\d)", "0", s)
    s = re.sub(r"(?<=\d)[lI](?=\d)", "1", s)
    s = re.sub(r"(?<=\d)[lI]", "1", s)
    s = re.sub(r"[lI](?=\d)", "1", s)
    s = s.replace(",", "")

    # Ignore Rs. decimal artifact
    s = re.sub(r"^\.", "", s)

    # Extract number
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        out["notes"].append("no_numeric_found")
        return out

    num = m.group(0)
    value = float(num)

    # Upper bound check
    if abs(value) > AMOUNT_UPPER_BOUND:
        out["notes"].append("amount_exceeds_upper_bound")
        return out

    out["value"] = value
    out["canonical"] = (currency + " " if currency else "") + f"{value:.2f}"
    out["confidence"] = "high"
    out["score"] = 0.9
    out["corrected"] = True

    return out


# ----------------------------------------------------------
# DATE NORMALIZER
# ----------------------------------------------------------
def normalize_date(raw):
    out = {
        "original": raw,
        "value": None,
        "canonical": None,
        "confidence": "low",
        "score": 0.0,
        "corrected": False,
        "notes": [],
    }

    if not raw:
        out["notes"].append("empty_input")
        return out

    s = str(raw).strip()

    # Fast ISO passthrough (YYYY-MM-DD)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        out["value"] = s
        out["canonical"] = s
        out["confidence"] = "high"
        out["score"] = 1.0
        return out

    # ---------- OCR FIXES ----------
    # Replace capital O or letter o between digits (2O24 -> 2024)
    s = re.sub(r"(?<=\d)[Oo](?=\d)", "0", s)

    # Also handle either side adjacency (digit-O OR O-digit)
    # but avoid changing text like "Oct" by limiting to cases where at least
    # one side is a digit (we already handled digit-digit above).
    s = re.sub(r"(?<=\d)[Oo]", "0", s)
    s = re.sub(r"[Oo](?=\d)", "0", s)

    # Replace lowercase L / uppercase I when they appear adjacent to digits (l -> 1)
    s = re.sub(r"(?<=\d)[lI](?=\d)", "1", s)
    s = re.sub(r"(?<=\d)[lI]", "1", s)
    s = re.sub(r"[lI](?=\d)", "1", s)

    # Common token-level OCR combos (case-insensitive)
    s = re.sub(r"(?i)lO", "10", s)  # lO, Lo -> 10
    s = re.sub(r"(?i)Ol", "01", s)  # Ol, oL -> 01
    s = re.sub(r"(?i)O1", "01", s)
    s = re.sub(r"(?i)1O", "10", s)

    # Remove weird non-printable characters that sometimes appear in OCR
    s = "".join(ch for ch in s if ord(ch) >= 32)

    # Normalize common separators to a single style (keep -, /, space, .)
    s = re.sub(r"[^\w\-/\. ]+", " ", s)

    # Collapse multiple separators/spaces
    s = re.sub(r"[\s\-/\.]+", " ", s).strip()

    # Now try parsing
    try:
        # Use dateutil parser; keep dayfirst=False to match earlier behavior
        dt = date_parser.parse(s, fuzzy=False, dayfirst=False)
        # simple sanity on year
        if 1900 <= dt.year <= datetime.utcnow().year + 2:
            iso = dt.date().isoformat()
            out["value"] = iso
            out["canonical"] = iso
            out["confidence"] = "high"
            out["score"] = 0.95
            out["corrected"] = True
            return out
    except Exception:
        # fall through to parse_failed
        pass

    out["notes"].append("parse_failed")
    return out



# ----------------------------------------------------------
# PHONE NORMALIZER
# ----------------------------------------------------------
def normalize_phone(raw, default_country="IN"):
    out = {
        "original": raw,
        "value": None,
        "canonical": None,
        "confidence": "low",
        "score": 0.0,
        "corrected": False,
        "notes": [],
    }

    if not raw:
        out["notes"].append("empty_input")
        return out

    s = re.sub(r"[^\d\+]", "", str(raw))
    digits = re.sub(r"[^\d]", "", s)

    if len(digits) < 7 or len(digits) > 15:
        out["notes"].append("invalid_length")
        return out

    # Indian 10-digit
    if len(digits) == 10 and digits[0] in "6789":
        out["value"] = digits
        out["canonical"] = "+91" + digits
        out["confidence"] = "high"
        out["score"] = 0.9
        return out

    # Indian with leading zero
    if len(digits) == 11 and digits.startswith("0"):
        out["value"] = digits[1:]
        out["canonical"] = "+91" + digits[1:]
        out["corrected"] = True
        out["confidence"] = "high"
        out["score"] = 0.85
        return out

    # Generic international
    out["value"] = digits
    out["canonical"] = "+" + digits
    out["confidence"] = "medium"
    out["score"] = 0.7
    return out


# ----------------------------------------------------------
# EMAIL NORMALIZER
# ----------------------------------------------------------
def normalize_email(raw):
    out = {
        "original": raw,
        "value": None,
        "canonical": None,
        "confidence": "low",
        "score": 0.0,
        "corrected": False,
        "notes": [],
    }

    if not raw:
        out["notes"].append("empty_input")
        return out

    s = str(raw).lower().replace(" ", "")

    if "(at)" in s:
        s = s.replace("(at)", "@")
        out["corrected"] = True

    if "@" not in s:
        out["notes"].append("missing_at_symbol")
        return out

    local, domain = s.split("@", 1)

    for bad, good in COMMON_DOMAIN_CORRECTIONS.items():
        if bad in domain:
            domain = good
            out["corrected"] = True

    candidate = f"{local}@{domain}"

    if re.match(r"^[a-z0-9_.+\-]+@[a-z0-9\-\.]+\.[a-z]{2,}$", candidate):
        out["value"] = candidate
        out["canonical"] = candidate
        out["confidence"] = "medium"
        out["score"] = 0.75
        return out

    out["notes"].append("failed_final_validation")
    return out


# ----------------------------------------------------------
# STRING NORMALIZER
# ----------------------------------------------------------
def normalize_string(raw, max_len=MAX_STRING_LEN):
    out = {
        "original": raw,
        "value": None,
        "confidence": "high",
        "score": 1.0,
        "corrected": False,
        "notes": [],
        "metadata": {},
    }

    if raw is None:
        out["notes"].append("empty_input")
        out["value"] = None
        out["confidence"] = "low"
        out["score"] = 0.0
        return out

    s = str(raw)

    # Unicode cleanup
    for k, v in LIGATURES_MAP.items():
        s = s.replace(k, v)

    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()

    if len(s) > max_len:
        s = s[:max_len]
        out["corrected"] = True
        out["notes"].append(f"truncated_to_{max_len}")

    out["value"] = s
    out["metadata"]["length"] = len(s)
    return out






# # claim_pipeline/processing/normalizer.py
# """
# Robust normalizer utilities for claims pipeline.

# Provides:
# - normalize_amount(amount_str) -> dict
# - normalize_date(date_str) -> dict
# - normalize_phone(phone_str) -> dict
# - normalize_email(email_str) -> dict
# - normalize_string(s) -> dict

# Each function returns a structured dict:
# {
#   "original": "<raw input>",
#   "value": <normalized value or None>,
#   "canonical": "<string representation or None>",
#   "confidence": "low"|"medium"|"high",
#   "score": float(0.0-1.0),
#   "corrected": bool,   # whether we applied an auto-correction
#   "notes": [ ... ],
#   "metadata": {...}    # optional (currency, detected_format, year, etc.)
# }
# """

# from __future__ import annotations
# import re
# import logging
# import unicodedata
# from datetime import datetime
# from dateutil import parser as date_parser
# from typing import Optional, Dict, Any, List

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


# # ---------- Configuration ----------
# MAX_STRING_LEN = 2000
# AMOUNT_UPPER_BOUND = 1e10  # 10 billion as practical upper bound for sanity-check
# YEAR_MIN = 1900
# YEAR_MAX = datetime.utcnow().year + 1

# COMMON_DOMAIN_CORRECTIONS = {
#     "gmai.com": "gmail.com",
#     "gmaiI.com": "gmail.com",
#     "gmai1.com": "gmail.com",
#     "gmial.com": "gmail.com",
#     "gmall.com": "gmail.com",
#     "gma1l.com": "gmail.com",
#     "yaho.com": "yahoo.com",
#     "yaho0.com": "yahoo.com",
#     "hotnail.com": "hotmail.com",
#     "hotmai.com": "hotmail.com",
#     "outllok.com": "outlook.com",
#     "outlok.com": "outlook.com",
#     "icloud.con": "icloud.com",
# }

# LIGATURES_MAP = {
#     "\ufb01": "fi",
#     "\ufb02": "fl",
#     "\u2013": "-",  # en dash
#     "\u2014": "-",  # em dash
#     "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'
# }

# # ---------- Helpers ----------
# def _score_label(score: float) -> str:
#     if score >= 0.85:
#         return "high"
#     if score >= 0.5:
#         return "medium"
#     return "low"


# def _clean_ocr_noise_num(s: str) -> str:
#     """
#     Attempt common OCR fixes for numbers:
#     - Replace common mistaken letters with digits where it makes sense
#     - Remove stray spaces
#     """
#     if not s:
#         return s
#     s = s.strip()
#     # common OCR confusions in numbers
#     s = s.replace(" ", "")
#     s = s.replace("O", "0").replace("o", "0")
#     s = s.replace("I", "1").replace("l", "1")  # careful: will affect text; used only for numeric strings
#     s = s.replace("S", "5")
#     s = s.replace("s", "5")
#     # remove currency words
#     s = re.sub(r"\b(INR|Rs\.?|Rs|USD|US\$)\b", "", s, flags=re.IGNORECASE)
#     # normalize multiple dots or commas
#     s = re.sub(r"[^\d\.,\-]", "", s)
#     # If multiple dots, keep last as decimal separator
#     if s.count(".") > 1:
#         parts = s.split(".")
#         s = "".join(parts[:-1]) + "." + parts[-1]
#     # Remove isolated commas incorrectly placed (e.g., '4, 50,0 00')
#     s = s.replace(",", "")
#     return s


# def _safe_float_from_string(s: str) -> Optional[float]:
#     try:
#         return float(s)
#     except Exception:
#         try:
#             # remove any non-digit except dot and minus, then parse
#             s2 = re.sub(r"[^\d\.\-]", "", s)
#             return float(s2) if s2 else None
#         except Exception:
#             return None


# def _normalize_unicode_text(s: str) -> str:
#     if s is None:
#         return s
#     # replace ligatures and common unicode punctuation
#     for k, v in LIGATURES_MAP.items():
#         s = s.replace(k, v)
#     s = unicodedata.normalize("NFKC", s)
#     # remove nulls
#     s = s.replace("\x00", "")
#     # unify newlines and whitespace
#     s = s.replace("\r\n", "\n").replace("\r", "\n")
#     s = re.sub(r"[ \t]{2,}", " ", s)
#     s = re.sub(r"\n{3,}", "\n\n", s)
#     return s.strip()


# # ---------- Normalizer Functions ----------

# def normalize_amount(amount_str: str) -> Dict[str, Any]:
#     """
#     Normalize amount strings to numeric value and canonical string.
#     Handles INR/US$ symbols, OCR noise, negative values, and basic sanity checks.
#     """
#     out = {
#         "original": amount_str,
#         "value": None,
#         "canonical": None,
#         "confidence": "low",
#         "score": 0.0,
#         "corrected": False,
#         "notes": [],
#         "metadata": {}
#     }
#     if not amount_str:
#         out["notes"].append("empty_input")
#         return out

#     s_raw = str(amount_str).strip()
#     s = s_raw

#     # Clean common OCR errors
#     s = s.replace("\u200b", "")  # zero-width
#     s = s.replace(" ", "")
#     s = s.replace("\u00A0", "")
#     # capture currency if present
#     currency = None
#     if re.search(r"\bINR\b|\bRs\b|₹", s_raw, re.IGNORECASE):
#         currency = "INR"
#     elif re.search(r"\bUSD\b|\bUS\$|\$", s_raw, re.IGNORECASE):
#         currency = "USD"
#     out["metadata"]["currency_hint"] = currency

#     # apply OCR noise fixes targeted for numeric content
#     s_fixed = _clean_ocr_noise_num(s_raw)
#     if s_fixed != s_raw:
#         out["corrected"] = True
#         out["notes"].append("ocr_numeric_fixes")
#     s = s_fixed

#     # extract a numeric substring candidate
#     # allow leading minus
#     m = re.search(r"-?\d+(\.\d+)?", s)
#     if not m:
#         # sometimes commas used wrongly: remove non digits and retry
#         s2 = re.sub(r"[^\d\.\-]", "", s)
#         m2 = re.search(r"-?\d+(\.\d+)?", s2)
#         if not m2:
#             out["notes"].append("no_numeric_found")
#             return out
#         else:
#             num_str = m2.group(0)
#     else:
#         num_str = m.group(0)

#     # final attempt to parse float
#     val = _safe_float_from_string(num_str)
#     if val is None:
#         out["notes"].append("parse_failed")
#         return out

#     # sanity checks
#     if abs(val) > AMOUNT_UPPER_BOUND:
#         out["notes"].append("amount_exceeds_upper_bound")
#         out["confidence"] = "low"
#         out["score"] = 0.1
#         out["value"] = None
#         return out

#     # success - canonical formatting with two decimals
#     out["value"] = float(val)
#     out["canonical"] = f"{float(val):.2f}"
#     out["metadata"]["detected_number"] = num_str
#     if currency:
#         out["metadata"]["currency_hint"] = currency
#         out["canonical"] = f"{currency} {out['canonical']}"
#     # scoring heuristics
#     score = 0.6
#     # higher score if currency symbol was present or commas existed
#     if currency or ("," in s_raw) or re.search(r"[₹\$]", s_raw):
#         score = 0.9
#     # reduce score if we applied heavy fixes
#     if out["corrected"]:
#         score -= 0.15
#     out["score"] = max(0.0, min(1.0, score))
#     out["confidence"] = _score_label(out["score"])
#     return out


# def normalize_date(date_str: str) -> Dict[str, Any]:
#     """
#     Normalize date-like strings to ISO YYYY-MM-DD.
#     Uses dateutil with preprocessing to handle OCR mistakes and ordinals.
#     """
#     out = {
#         "original": date_str,
#         "value": None,
#         "canonical": None,
#         "confidence": "low",
#         "score": 0.0,
#         "corrected": False,
#         "notes": [],
#         "metadata": {}
#     }
#     if not date_str:
#         out["notes"].append("empty_input")
#         return out

#     s_raw = str(date_str).strip()
#     s = s_raw

#     # common OCR fixes: replace I -> 1 when likely in year context, O -> 0
#     s = re.sub(r"(\d)O(\d)", r"\10\2", s)  # 1O2 -> 102
#     s = re.sub(r"([^\d]|^)[OI]([^\d])", r"\11\2", s)  # rough attempt
#     s = re.sub(r"(\d)I(\d)", r"\11\2", s)
#     # remove ordinal suffixes (1st, 2nd, 12th)
#     s = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)
#     s = s.replace(".", "-")  # sometimes dots used as separators in OCR
#     s = s.strip()
#     if s != s_raw:
#         out["corrected"] = True
#         out["notes"].append("ocr_date_fixes")

#     # Try multiple parse strategies
#     parse_attempts = []
#     parsed_date = None
#     for dayfirst in (False, True):
#         try:
#             dt = date_parser.parse(s, dayfirst=dayfirst, fuzzy=True, default=datetime(1900,1,1))
#             # validity checks
#             if YEAR_MIN <= dt.year <= YEAR_MAX:
#                 parsed_date = dt
#                 out["metadata"]["parsed_dayfirst"] = dayfirst
#                 break
#             else:
#                 out["notes"].append(f"year_out_of_range:{dt.year}")
#         except Exception as e:
#             parse_attempts.append(str(e))
#             continue

#     if not parsed_date:
#         out["notes"].append("parse_failed")
#         out["notes"].extend(parse_attempts)
#         return out

#     # canonical ISO
#     iso = parsed_date.date().isoformat()
#     out["value"] = iso
#     out["canonical"] = iso
#     # scoring heuristic
#     score = 0.7
#     # higher if format obvious (YYYY-MM-DD)
#     if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}$", s_raw.strip()):
#         score = 0.95
#     if out["corrected"]:
#         score -= 0.1
#     out["score"] = max(0.0, min(1.0, score))
#     out["confidence"] = _score_label(out["score"])
#     out["metadata"]["year"] = parsed_date.year
#     return out


# def normalize_phone(phone_str: str, default_country: str = "IN") -> Dict[str, Any]:
#     """
#     Normalize phone numbers. Primary heuristics for Indian numbers:
#     - If 10 digits -> +91XXXXXXXXXX
#     - If begins with 0 and then 10 digits -> +91...
#     - If digits length between 7 and 15, accept with country prefix if present
#     Returns E.164-like canonical as string or None.
#     """
#     out = {
#         "original": phone_str,
#         "value": None,
#         "canonical": None,
#         "confidence": "low",
#         "score": 0.0,
#         "corrected": False,
#         "notes": [],
#         "metadata": {}
#     }
#     if not phone_str:
#         out["notes"].append("empty_input")
#         return out

#     s_raw = str(phone_str)
#     # remove common separators but keep leading +
#     s = s_raw.strip()
#     s = s.replace("(", "").replace(")", "").replace(".", " ").replace("/", " ")
#     s = re.sub(r"[^\d\+]", "", s)

#     # strip leading '+' for digit count
#     digits = re.sub(r"[^\d]", "", s)

#     if len(digits) < 7 or len(digits) > 15:
#         out["notes"].append("invalid_length")
#         out["score"] = 0.05
#         out["confidence"] = "low"
#         return out

#     # Common Indian mobile heuristic: 10 digits starting with 6-9
#     if len(digits) == 10 and digits[0] in "6789":
#         canonical = "+91" + digits
#         out["value"] = digits
#         out["canonical"] = canonical
#         out["score"] = 0.9
#         out["confidence"] = "high"
#         out["metadata"]["detected_country"] = "IN"
#         return out

#     # If starts with '0' followed by 10 digits
#     if len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
#         canonical = "+91" + digits[1:]
#         out["value"] = digits[1:]
#         out["canonical"] = canonical
#         out["score"] = 0.85
#         out["confidence"] = "high"
#         out["corrected"] = True
#         out["notes"].append("leading_zero_removed_assumed_IN")
#         return out

#     # If already contains country code (starts with country code digits length 11-15)
#     if len(digits) >= 11 and len(digits) <= 15:
#         canonical = "+" + digits
#         out["value"] = digits
#         out["canonical"] = canonical
#         out["score"] = 0.7
#         out["confidence"] = "medium"
#         return out

#     # Fallback: return digits with low confidence
#     out["value"] = digits
#     out["canonical"] = ("+" + digits) if not s.startswith("+") else s
#     out["score"] = 0.3
#     out["confidence"] = "low"
#     out["notes"].append("fallback_numeric_only")
#     return out


# def normalize_email(email_str: str) -> Dict[str, Any]:
#     """
#     Normalize and attempt to autocorrect common OCR errors in emails.
#     Uses conservative corrections and reports confidence.
#     """
#     out = {
#         "original": email_str,
#         "value": None,
#         "canonical": None,
#         "confidence": "low",
#         "score": 0.0,
#         "corrected": False,
#         "notes": [],
#         "metadata": {}
#     }
#     if not email_str:
#         out["notes"].append("empty_input")
#         return out

#     s_raw = str(email_str).strip()
#     s = s_raw.replace(" ", "").replace("\u200b", "")
#     # Try simple lowercasing and basic cleanup
#     s = s.strip().lower()

#     # Remove leading/trailing dots in local or domain
#     s = re.sub(r"(^\.)|(\.$)", "", s)

#     # Basic validation
#     if "@" not in s:
#         out["notes"].append("missing_at_symbol")
#         # maybe OCR turned '@' to '(at)' or 'at' -> try to fix common patterns
#         s_guess = re.sub(r"\s*\(at\)\s*|\s+at\s+", "@", s)
#         if "@" in s_guess:
#             s = s_guess
#             out["corrected"] = True
#             out["notes"].append("replaced_at_token")
#         else:
#             out["score"] = 0.0
#             out["confidence"] = "low"
#             return out

#     parts = s.split("@")
#     if len(parts) != 2:
#         out["notes"].append("invalid_structure")
#         out["score"] = 0.0
#         return out

#     local, domain = parts[0], parts[1]

#     # Fix common OCR substitutions in domain
#     for bad, good in COMMON_DOMAIN_CORRECTIONS.items():
#         if bad in domain:
#             domain = domain.replace(bad, good)
#             out["corrected"] = True
#             out["notes"].append(f"domain_fixed:{bad}->{good}")

#     # common OCR: replace I (capital i) with l in domain/ local sometimes
#     domain = domain.replace("i", "i")  # noop to avoid over-correction; keep conservative

#     # Remove leading/trailing dots and multiple dots
#     domain = re.sub(r"\.{2,}", ".", domain)
#     domain = domain.strip(".-")

#     # Validate domain structure
#     if not re.match(r"^[a-z0-9\-\.]+\.[a-z]{2,}$", domain):
#         # attempt small corrections, e.g. gmai -> gmail.com
#         for bad, good in COMMON_DOMAIN_CORRECTIONS.items():
#             if bad.split(".")[0] in domain:
#                 domain = good
#                 out["corrected"] = True
#                 out["notes"].append(f"domain_autocorrect:{good}")
#                 break

#     # validate local part more permissively but remove leading/trailing dots
#     local = local.strip(".")
#     local = re.sub(r"[^\w\.\+\-]", "", local)

#     candidate = f"{local}@{domain}"
#     # Final validation
#     if re.match(r"^[a-z0-9_.+\-]+@[a-z0-9\-\.]+\.[a-z]{2,}$", candidate):
#         out["value"] = candidate
#         out["canonical"] = candidate
#         # scoring heuristics
#         score = 0.7
#         if out["corrected"]:
#             score -= 0.15
#         # boost if domain well-known
#         if domain.endswith(("gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com")):
#             score += 0.15
#         out["score"] = max(0.0, min(1.0, score))
#         out["confidence"] = _score_label(out["score"])
#     else:
#         out["notes"].append("failed_final_validation")
#         out["score"] = 0.0
#         out["confidence"] = "low"

#     return out


# def normalize_string(s: str, max_len: int = MAX_STRING_LEN) -> Dict[str, Any]:
#     """
#     Normalize arbitrary text:
#     - remove ligatures, normalize unicode
#     - collapse whitespace
#     - trim to max_len
#     Returns structured dict with original + cleaned.
#     """
#     out = {
#         "original": s,
#         "value": None,
#         "confidence": "high",
#         "score": 1.0,
#         "corrected": False,
#         "notes": [],
#         "metadata": {}
#     }
#     if s is None:
#         out["notes"].append("empty_input")
#         out["value"] = None
#         out["score"] = 0.0
#         out["confidence"] = "low"
#         return out

#     raw = str(s)
#     cleaned = _normalize_unicode_text(raw)
#     if cleaned != raw:
#         out["corrected"] = True
#         out["notes"].append("unicode_normalized")

#     # collapse multi-space to single
#     cleaned = re.sub(r"[ \t]+", " ", cleaned)
#     # truncate
#     if len(cleaned) > max_len:
#         out["notes"].append(f"truncated_to_{max_len}")
#         cleaned = cleaned[:max_len]
#         out["corrected"] = True

#     out["value"] = cleaned
#     out["metadata"]["length"] = len(cleaned)
#     out["score"] = 0.95
#     out["confidence"] = _score_label(out["score"])
#     return out
