# claim_pipeline/extraction/pdf_extractor.py

import os
import fitz  # PyMuPDF
import hashlib
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _compute_pdf_sha(pdf_path: str) -> str:
    try:
        with open(pdf_path, "rb") as f:
            return _sha256_bytes(f.read())
    except Exception as e:
        logger.warning(f"[pdf_extractor] Could not compute PDF SHA256: {e}")
        return ""


def _heuristic_table_detect(blocks: List[tuple]) -> bool:
    """
    Simple heuristic to detect table-like layout:
    - If there are many blocks
    - If many blocks align vertically (similar x0 values) across multiple rows
    This is a light-weight signal, not a perfect detector.
    """
    if not blocks or len(blocks) < 6:
        return False

    # collect rounded x0 positions
    x0_positions = [round(b[0]) for b in blocks if len(b) >= 5]
    # frequency of x0 positions
    freq = {}
    for x in x0_positions:
        freq[x] = freq.get(x, 0) + 1

    # table-like if multiple columns (>=3) with repeated x positions
    repeated = sum(1 for v in freq.values() if v >= 2)
    return repeated >= 3


def _page_confidence(text: str, num_blocks: int, has_table: bool) -> float:
    """
    Heuristic confidence:
    - longer text -> higher confidence
    - more blocks -> slightly higher
    - table presence reduces extraction confidence for plain linearized text
    """
    base = 0.0
    length = len(text or "")
    if length > 2000:
        base = 0.95
    elif length > 800:
        base = 0.85
    elif length > 200:
        base = 0.6
    elif length > 50:
        base = 0.4
    else:
        base = 0.2

    # adjust by blocks
    base += min(0.05, num_blocks * 0.01)

    # penalty if table detected
    if has_table:
        base -= 0.2

    # clamp
    if base < 0.05:
        base = 0.05
    if base > 0.99:
        base = 0.99

    return round(base, 2)


def extract_text_from_pdf(pdf_path: str, fallback_to_ocr: bool = True) -> Dict[str, Any]:
    """
    Extract text and rich metadata from a machine-readable PDF.

    Returns a dict:
    {
        "text": "<full text>",
        "pages": [
            {
                "page_no": 1,
                "text": "...",
                "blocks": [ (x0,y0,x1,y1, "text", block_no), ... ],
                "table_detected": bool,
                "sha256": "<page-sha>",
                "confidence": 0.85,
                "notes": [ ... ]
            }, ...
        ],
        "num_pages": N,
        "pdf_sha256": "...",
        "fallback_used": True/False
    }

    Notes:
    - This function is defensive: it skips corrupt pages and continues.
    - If the entire PDF yields empty text, it will optionally attempt to call an OCR fallback
      function named `ocr_pdf_scanned` from `claim_pipeline.extraction.ocr_extractor`
      (if available). If not available, it will log and return empty extraction.
    """
    result: Dict[str, Any] = {"text": "", "pages": [], "num_pages": 0, "pdf_sha256": "", "fallback_used": False}
    pdf_sha = _compute_pdf_sha(pdf_path)
    result["pdf_sha256"] = pdf_sha

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"[pdf_extractor] Failed to open PDF {pdf_path}: {e}")
        return result

    full_text_parts = []
    page_count = len(doc)

    for i in range(page_count):
        page_no = i + 1
        page_entry = {
            "page_no": page_no,
            "text": "",
            "blocks": [],
            "table_detected": False,
            "sha256": "",
            "confidence": 0.0,
            "notes": []
        }

        try:
            page = doc.load_page(i)

            # Attempt to get block tuples
            # get_text("blocks") returns list of tuples (x0,y0,x1,y1, "text", block_no)
            try:
                blocks = page.get_text("blocks")
            except Exception as e:
                logger.warning(f"[pdf_extractor] page.get_text('blocks') failed on page {page_no}: {e}")
                blocks = []

            # Sort blocks top-to-bottom, left-to-right
            try:
                blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0])) if blocks else []
            except Exception:
                blocks_sorted = blocks

            # Join text from blocks preserving order
            page_text = "\n".join(b[4].strip() for b in blocks_sorted if len(b) >= 5 and b[4].strip())
            page_entry["text"] = page_text
            page_entry["blocks"] = blocks_sorted

            # Heuristic table detection
            table_detected = _heuristic_table_detect(blocks_sorted)
            page_entry["table_detected"] = table_detected

            # Page-level SHA: prefer text bytes; if empty, fallback to page image bytes
            if page_text.strip():
                page_entry["sha256"] = _sha256_bytes(page_text.encode("utf-8"))
            else:
                # render page pixmap as fallback for hashing if text missing
                try:
                    pix = page.get_pixmap()
                    page_bytes = pix.tobytes()
                    page_entry["sha256"] = _sha256_bytes(page_bytes)
                    page_entry["notes"].append("No selectable text; hashed page image.")
                except Exception as e:
                    page_entry["sha256"] = ""
                    page_entry["notes"].append(f"Failed to generate page image for hash: {e}")

            # Confidence heuristic
            page_entry["confidence"] = _page_confidence(page_entry["text"], len(blocks_sorted), table_detected)

            # If page empty, add a note / log
            if not page_entry["text"].strip():
                msg = f"Page {page_no} has no extracted text."
                logger.info(f"[pdf_extractor] {msg}")
                page_entry["notes"].append(msg)

            full_text_parts.append(page_entry["text"])
            result["pages"].append(page_entry)

        except Exception as e:
            # Per-page error handling: log, note, and continue
            logger.error(f"[pdf_extractor] Error processing page {page_no} of {pdf_path}: {e}")
            page_entry["notes"].append(f"Error processing page: {e}")
            page_entry["confidence"] = 0.05
            result["pages"].append(page_entry)
            continue

    # Combine full text
    combined = "\n".join([p["text"] for p in result["pages"] if p.get("text")])
    result["text"] = combined.strip()
    result["num_pages"] = len(result["pages"])

    # If result is essentially empty and fallback is allowed, attempt OCR fallback
    if (not result["text"].strip()) and fallback_to_ocr:
        logger.info(f"[pdf_extractor] No machine text extracted from {pdf_path}. Attempting OCR fallback.")
        result["fallback_used"] = True
        try:
            # Try to import a known OCR fallback function
            from claim_pipeline.extraction.ocr_extractor import ocr_pdf_scanned  # type: ignore

            # ocr_pdf_scanned should accept (pdf_path) and return aggregated text and optionally meta
            try:
                ocr_output = ocr_pdf_scanned(pdf_path)
                if isinstance(ocr_output, dict):
                    # Expect ocr_output to contain keys 'text' and maybe 'pages'
                    ocr_text = ocr_output.get("text", "")
                else:
                    # if it's just text
                    ocr_text = str(ocr_output or "")
                result["text"] = ocr_text or ""
                logger.info(f"[pdf_extractor] OCR fallback produced text (len={len(result['text'])}).")
            except Exception as e:
                logger.error(f"[pdf_extractor] OCR fallback failed on {pdf_path}: {e}")

        except Exception:
            # No OCR module available — log and continue
            logger.warning("[pdf_extractor] OCR fallback not available (ocr_extractor.ocr_pdf_scanned not found).")

    return result