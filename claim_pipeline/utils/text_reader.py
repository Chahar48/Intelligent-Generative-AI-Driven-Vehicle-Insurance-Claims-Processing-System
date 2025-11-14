# claim_pipeline/utils/text_reader.py
"""
Robust text reader utility for claim_pipeline.

Functions:
- read_text_file(...) -> returns dict with keys:
    {
      "text": str,
      "lines": Optional[List[str]],
      "encoding": str,
      "size": int,
      "sha256": str
    }

- read_text_file_str(...) -> backward-compatible wrapper returning only the text string.
"""

import os
import logging
import hashlib
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------------
# Helpers
# -------------------------
def compute_sha256_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"[text_reader] Failed to compute sha256 for {path}: {e}")
        return ""


# -------------------------
# Main function
# -------------------------
def read_text_file(
    file_path: str,
    *,
    max_size: int = 5 * 1024 * 1024,
    return_lines: bool = False,
    encodings_to_try: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Read a plain text file safely and return structured data.

    Parameters
    ----------
    file_path : str
        Path to the text file.
    max_size : int, optional
        Maximum allowed file size in bytes (default 5 MB). Larger files will be rejected.
    return_lines : bool, optional
        If True, also return 'lines' (list of normalized lines).
    encodings_to_try : Optional[List[str]]
        List of encodings to attempt in order. If None, uses sensible defaults.

    Returns
    -------
    dict
        {
            "text": "<normalized text>",
            "lines": [ ... ]  # present only if return_lines True
            "encoding": "<encoding used>",
            "size": <bytes>,
            "sha256": "<hex>",
        }
    """
    if encodings_to_try is None:
        # sensible fallback order
        encodings_to_try = ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"]

    result: Dict[str, Any] = {
        "text": "",
        "encoding": None,
        "size": 0,
        "sha256": "",
    }

    # Basic checks
    if not os.path.isfile(file_path):
        logger.error(f"[text_reader] File not found: {file_path}")
        return result

    try:
        size = os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"[text_reader] Could not get size for {file_path}: {e}")
        return result

    result["size"] = size

    if size > max_size:
        logger.warning(f"[text_reader] File {file_path} exceeds max_size ({size} bytes > {max_size}). Skipping read.")
        return result

    # Compute sha256 for audit
    result["sha256"] = compute_sha256_file(file_path)

    # Try different encodings
    raw_text = None
    used_encoding = None
    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc, errors="strict") as f:
                raw_text = f.read()
            used_encoding = enc
            logger.info(f"[text_reader] Successfully read {file_path} with encoding '{enc}'")
            break
        except UnicodeDecodeError:
            logger.debug(f"[text_reader] UnicodeDecodeError with encoding {enc} for file {file_path}; trying next.")
            continue
        except Exception as e:
            logger.warning(f"[text_reader] Error reading {file_path} with encoding {enc}: {e}")
            continue

    if raw_text is None:
        # As a last resort, try reading with 'latin-1' permissive mode to salvage text
        try:
            with open(file_path, "r", encoding="latin-1", errors="replace") as f:
                raw_text = f.read()
            used_encoding = "latin-1(replace)"
            logger.info(f"[text_reader] Read {file_path} using latin-1 with replacement.")
        except Exception as e:
            logger.error(f"[text_reader] Failed to read file with fallback encoding for {file_path}: {e}")
            return result

    # Normalize BOM
    if raw_text.startswith("\ufeff"):
        raw_text = raw_text.lstrip("\ufeff")
        logger.debug(f"[text_reader] Stripped BOM from {file_path}")

    # Remove null characters and normalize newlines
    try:
        cleaned = raw_text.replace("\x00", "")
        # Normalize different newline styles to '\n'
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        # Trim leading/trailing whitespace
        cleaned = cleaned.strip()
    except Exception as e:
        logger.warning(f"[text_reader] Error cleaning text for {file_path}: {e}")
        cleaned = raw_text

    result["text"] = cleaned
    result["encoding"] = used_encoding

    # Optionally return lines (non-empty)
    if return_lines:
        # split by '\n' and strip each line
        lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
        result["lines"] = lines

    return result


# -------------------------
# Backward-compatible wrapper
# -------------------------
def read_text_file_str(file_path: str) -> str:
    """
    Backward-compatible wrapper that returns only the text string (or empty string on error).
    """
    res = read_text_file(file_path, return_lines=False)
    return res.get("text", "") or ""
