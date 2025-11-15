"""
PII Redactor — simplified, stable, and pipeline-safe.

Redacts:
- Emails
- Phones
- Claim/Policy IDs
- Names
- Free-text blobs
- Nested dict/list structures
"""

import re
from typing import Any, Dict

# ---------------------------
# Regex Patterns
# ---------------------------
_EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]{1,64})@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s\.]{6,}\d)")
_CLAIM_POLICY_RE = re.compile(
    r"\b(?P<prefix>CLM|CLAIM|PN|POLICY|POL)\b(?P<sep>[-:\s]*)?(?P<id>[A-Z0-9\-_\/]+)",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"\b([A-Z][a-z]{1,}\s+[A-Z][a-z]{1,})\b")


# ---------------------------
# Email
# ---------------------------
def redact_email(email: str) -> str:
    if not email:
        return email
    m = _EMAIL_RE.search(email)
    if not m:
        return email

    user, domain = m.group(1), m.group(2)
    keep = max(1, min(3, len(user) // 2))
    masked = user[:keep] + "*" * (len(user) - keep)
    return f"{masked}@{domain}"


# ---------------------------
# Phone
# ---------------------------
def redact_phone(phone: str) -> str:
    if not phone:
        return phone
    digits = re.sub(r"[^\d]", "", phone)
    if not digits:
        return phone

    # Use last 10 digits (mobile number)
    local = digits[-10:] if len(digits) > 10 else digits

    if len(local) <= 4:
        return "*" * len(local)

    return f"{local[:2]}{'*'*(len(local)-4)}{local[-2:]}"


# ---------------------------
# Claim/Policy IDs
# ---------------------------
def redact_claim_or_policy_id(text: str) -> str:
    if not text:
        return text

    def _mask(m):
        prefix, sep, val = m.group("prefix"), m.group("sep"), m.group("id")
        if len(val) <= 3:
            masked = val[0] + "*" * (len(val)-1)
        else:
            masked = val[:2] + "*" * (len(val)-3) + val[-1]
        return f"{prefix}{sep}{masked}"

    return _CLAIM_POLICY_RE.sub(_mask, text)


# ---------------------------
# Names
# ---------------------------
def redact_name(name: str) -> str:
    if not name:
        return name
    parts = name.split()
    if len(parts) < 2:
        return parts[0][0] + "***"

    first, last = parts[0], parts[-1]
    return f"{first} {last[0]}***"


# ---------------------------
# Block Redaction
# ---------------------------
def redact_text(text: str) -> str:
    if not text:
        return text

    out = text
    out = _EMAIL_RE.sub(lambda m: redact_email(m.group(0)), out)
    out = _PHONE_RE.sub(lambda m: redact_phone(m.group(0)), out)
    out = redact_claim_or_policy_id(out)
    out = _NAME_RE.sub(lambda m: redact_name(m.group(0)), out)

    return out


# ---------------------------
# Recursive Dict Redactor
# ---------------------------
def redact_dict(obj: Any, keys_to_keep: Dict[str, bool] = None) -> Any:
    if keys_to_keep is None:
        keys_to_keep = {}

    if isinstance(obj, dict):
        return {
            k: (
                v if keys_to_keep.get(k, False)
                else redact_dict(v, keys_to_keep)
            )
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [redact_dict(x, keys_to_keep) for x in obj]

    if isinstance(obj, str):
        return redact_text(obj)

    return obj


# # claim_pipeline/security/pii_redactor.py
# """
# PII Redactor utilities.

# Provides:
# - redact_email()
# - redact_phone()
# - redact_claim_or_policy_id()
# - redact_name()
# - redact_text()  # general purpose
# - redact_dict()  # recursively redact values in dict/list

# Designed for JSON-based pipeline (works with extracted fields, logs, audit events).
# """

# import re
# from typing import Any, Dict

# # Regexes
# _EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]{1,64})@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
# # Match phone-like sequences; we'll pass the whole match into redact_phone()
# _PHONE_RE = re.compile(r"(\+?\d[\d\-\s\.]{6,}\d)")
# # Capture prefix (CLM/CLAIM/PN/POLICY/POL), optional separator, then id token
# _CLAIM_POLICY_RE = re.compile(
#     r"\b(?P<prefix>CLM|CLAIM|PN|POLICY|POL)\b(?P<sep>[-:\s]*)?(?P<id>[A-Z0-9\-_\/]+)",
#     re.IGNORECASE,
# )
# _NAME_RE = re.compile(r"\b([A-Z][a-z]{1,}\s+[A-Z][a-z]{1,})\b")  # simple two-word name


# def redact_email(email: str) -> str:
#     if not email:
#         return email
#     m = _EMAIL_RE.search(email)
#     if not m:
#         return email
#     user, domain = m.group(1), m.group(2)
#     # keep 1-3 characters, mask the rest, ensure at least one star for tiny usernames
#     if len(user) <= 2:
#         u = user[0] + "*"
#     else:
#         keep = max(1, min(3, len(user) // 2))
#         u = user[:keep] + "*" * (len(user) - keep)
#     return f"{u}@{domain}"


# def redact_phone(phone: str) -> str:
#     """
#     Mask phone numbers while keeping a small prefix and suffix for recognizability.
#     Heuristics:
#       - Use the last 10 digits as the 'local' number if available (typical mobile length).
#       - Preserve 2 digits at front and 2 digits at back of local number; mask the rest.
#       - If number is <=4 digits -> mask entirely.
#       - Returns masked local-number style string (no country code returned).
#     """
#     if not phone:
#         return phone
#     # keep only digits
#     digits = re.sub(r"[^\d]", "", phone)
#     if not digits:
#         return phone

#     # If very short, mask everything
#     if len(digits) <= 4:
#         return "*" * len(digits)

#     # Use last 10 digits as local number if length > 10 (handles country codes)
#     if len(digits) > 10:
#         local = digits[-10:]
#     else:
#         local = digits

#     keep_front = 2
#     keep_back = 2
#     if len(local) <= (keep_front + keep_back):
#         # fallback: keep first and last char if possible
#         if len(local) <= 2:
#             return "*" * len(local)
#         return local[0] + ("*" * (len(local) - 2)) + local[-1]

#     masked = f"{local[:keep_front]}{'*' * (len(local) - keep_front - keep_back)}{local[-keep_back:]}"
#     return masked


# def redact_claim_or_policy_id(text: str) -> str:
#     if not text:
#         return text

#     def _repl(m):
#         prefix = m.group("prefix") or ""
#         sep = m.group("sep") or ""
#         val = m.group("id") or ""
#         # mask interior of id, keep first 2 and last 1, ensure at least one star
#         if len(val) <= 3:
#             masked = val[0] + "*" * (max(1, len(val) - 1))
#         else:
#             masked = val[:2] + "*" * (max(1, len(val) - 3)) + (val[-1] if len(val) > 2 else "")
#         return f"{prefix}{sep}{masked}"

#     return _CLAIM_POLICY_RE.sub(_repl, text)


# def redact_name(name: str) -> str:
#     if not name:
#         return name
#     parts = name.split()
#     if len(parts) == 1:
#         # keep first char, mask the rest; ensure at least 3 stars for consistency
#         rest_mask = "*" * max(3, len(parts[0]) - 1)
#         return parts[0][0] + rest_mask
#     first = parts[0]
#     last = parts[-1]
#     # keep first letter of last name, mask remaining - ensure at least 3 stars
#     last_masked = last[0] + "*" * max(3, len(last) - 1)
#     return f"{first} {last_masked}"


# def redact_text(text: str) -> str:
#     """
#     Apply a sequence of redactions on a free text blob.
#     Order matters: emails, phone, IDs, names.
#     """
#     if not text:
#         return text
#     out = text

#     # Emails (replace every email match with its masked version)
#     out = _EMAIL_RE.sub(lambda m: redact_email(m.group(0)), out)

#     # Phones: feed the whole matched substring to redact_phone()
#     out = _PHONE_RE.sub(lambda m: redact_phone(m.group(0)), out)

#     # claim / policy ids
#     out = redact_claim_or_policy_id(out)

#     # names (best-effort)
#     out = _NAME_RE.sub(lambda m: redact_name(m.group(0)), out)

#     return out


# def redact_dict(obj: Any, keys_to_keep: Dict[str, bool] = None) -> Any:
#     """
#     Recursively redact PII in dict/list/primitive types.
#     keys_to_keep: optional mapping of field names that should NOT be redacted (True means keep)
#     """
#     if keys_to_keep is None:
#         keys_to_keep = {}

#     if isinstance(obj, dict):
#         out = {}
#         for k, v in obj.items():
#             # honor keys_to_keep exactly (use original key)
#             if keys_to_keep.get(k, False):
#                 out[k] = v
#                 continue

#             kl = k.lower()
#             # id fields
#             if kl in ("claim_id", "policy_no", "policy", "policy_no"):
#                 if isinstance(v, str):
#                     out[k] = redact_claim_or_policy_id(v)
#                 else:
#                     out[k] = v
#             elif kl in ("email", "email_address") and isinstance(v, str):
#                 out[k] = redact_email(v)
#             elif kl in ("phone", "mobile", "contact") and isinstance(v, str):
#                 out[k] = redact_phone(v)
#             elif isinstance(v, (dict, list)):
#                 out[k] = redact_dict(v, keys_to_keep)
#             elif isinstance(v, str):
#                 # best-effort redact inside generic strings
#                 out[k] = redact_text(v)
#             else:
#                 out[k] = v
#         return out
#     elif isinstance(obj, list):
#         return [redact_dict(x, keys_to_keep) for x in obj]
#     elif isinstance(obj, str):
#         return redact_text(obj)
#     else:
#         return obj




# # claim_pipeline/security/pii_redactor.py
# """
# PII Redactor utilities.

# Provides:
# - redact_email()
# - redact_phone()
# - redact_claim_or_policy_id()
# - redact_name()
# - redact_text()  # general purpose
# - redact_dict()  # recursively redact values in dict/list

# Designed for JSON-based pipeline (works with extracted fields, logs, audit events).
# """

# import re
# from typing import Any, Dict

# # Regexes
# _EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]{1,64})@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
# _PHONE_RE = re.compile(r"(\+?\d{1,3}[-\s\.]?)?(\d{2,4})[-\s\.]?(\d{2,4})[-\s\.]?(\d{2,4})")
# _CLAIM_POLICY_RE = re.compile(r"\b(?:CLM|CLAIM|PN|POLICY|POL)\b[-:\s]*([A-Z0-9\-_/]+)", re.IGNORECASE)
# _NAME_RE = re.compile(r"\b([A-Z][a-z]{1,}\s+[A-Z][a-z]{1,})\b")  # simple two-word name

# def redact_email(email: str) -> str:
#     if not email:
#         return email
#     m = _EMAIL_RE.search(email)
#     if not m:
#         return email
#     user, domain = m.group(1), m.group(2)
#     if len(user) <= 2:
#         u = user[0] + "*"
#     else:
#         keep = max(1, min(3, len(user)//2))
#         u = user[:keep] + "*" * (len(user)-keep)
#     return f"{u}@{domain}"

# def redact_phone(phone: str) -> str:
#     if not phone:
#         return phone
#     digits = re.sub(r"[^\d]", "", phone)
#     if len(digits) <= 4:
#         return "*" * len(digits)
#     keep_front = 2
#     keep_back = 2
#     return f"{digits[:keep_front]}{'*'*(len(digits)-keep_front-keep_back)}{digits[-keep_back:]}"

# def redact_claim_or_policy_id(text: str) -> str:
#     if not text:
#         return text
#     def _repl(m):
#         val = m.group(1)
#         masked = val[:2] + "*"*(max(1, len(val)-3)) + (val[-1] if len(val) > 2 else "")
#         return m.group(0).replace(val, masked)
#     return _CLAIM_POLICY_RE.sub(_repl, text)

# def redact_name(name: str) -> str:
#     if not name:
#         return name
#     # mask last name except first char
#     parts = name.split()
#     if len(parts) == 1:
#         return parts[0][0] + "*"*(len(parts[0])-1)
#     first = parts[0]
#     last = parts[-1]
#     last_masked = last[0] + "*"*(max(1, len(last)-1))
#     return f"{first} {last_masked}"

# def redact_text(text: str) -> str:
#     """
#     Apply a sequence of redactions on a free text blob.
#     Order matters: emails, phone, IDs, names.
#     """
#     if not text:
#         return text
#     out = text
#     # Emails
#     out = _EMAIL_RE.sub(lambda m: redact_email(m.group(0)), out)
#     # Phones
#     out = _PHONE_RE.sub(lambda m: redact_phone("".join(m.groups() or [])), out)
#     # claim / policy ids
#     out = redact_claim_or_policy_id(out)
#     # names (best-effort)
#     out = _NAME_RE.sub(lambda m: redact_name(m.group(0)), out)
#     return out

# def redact_dict(obj: Any, keys_to_keep: Dict[str, bool] = None) -> Any:
#     """
#     Recursively redact PII in dict/list/primitive types.
#     keys_to_keep: optional mapping of field names that should NOT be redacted (True means keep)
#     """
#     if keys_to_keep is None:
#         keys_to_keep = {}

#     if isinstance(obj, dict):
#         out = {}
#         for k, v in obj.items():
#             if k.lower() in ("claim_id", "policy_no", "policy", "policy_no") and not keys_to_keep.get(k, False):
#                 # Mask id fields specifically
#                 if isinstance(v, str):
#                     out[k] = redact_claim_or_policy_id(v)
#                 else:
#                     out[k] = v
#             elif k.lower() in ("email", "email_address") and isinstance(v, str) and not keys_to_keep.get(k, False):
#                 out[k] = redact_email(v)
#             elif k.lower() in ("phone", "mobile", "contact") and isinstance(v, str) and not keys_to_keep.get(k, False):
#                 out[k] = redact_phone(v)
#             elif isinstance(v, (dict, list)):
#                 out[k] = redact_dict(v, keys_to_keep)
#             elif isinstance(v, str):
#                 # best-effort redact inside generic strings
#                 out[k] = redact_text(v)
#             else:
#                 out[k] = v
#         return out
#     elif isinstance(obj, list):
#         return [redact_dict(x, keys_to_keep) for x in obj]
#     elif isinstance(obj, str):
#         return redact_text(obj)
#     else:
#         return obj
