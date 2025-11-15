# claim_pipeline/ingestion/file_router.py

import os
import fitz  # PyMuPDF
from PIL import Image

# -------------------------------
# Simple Helpers
# -------------------------------

def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def pdf_has_text_layer(pdf_path: str, sample_pages: int = 2) -> bool:
    """Check if the first few pages have extractable text."""
    try:
        doc = fitz.open(pdf_path)
        for i in range(min(sample_pages, len(doc))):
            page = doc.load_page(i)
            text = page.get_text("text")
            if text.strip():
                return True
        return False
    except Exception:
        return False


# -------------------------------
# Main routing function
# -------------------------------

def detect_file_type(file_path: str) -> dict:
    """
    Simple file router.
    Always returns a dictionary with fixed structure:
    {
        "path": ...,
        "type": ...,
        "processor": ...,
        "notes": ...
    }
    """
    ext = os.path.splitext(file_path)[-1].lower()

    # -------------------------------
    # PDF
    # -------------------------------
    if ext == ".pdf":
        has_text = pdf_has_text_layer(file_path)
        processor = "pdf_text" if has_text else "pdf_scanned"

        return {
            "path": file_path,
            "type": "pdf",
            "processor": processor,
            "notes": "PDF with text layer" if has_text else "Scanned PDF (no text layer)"
        }

    # -------------------------------
    # IMAGE FILES
    # -------------------------------
    if ext in [".png", ".jpg", ".jpeg"]:
        try:
            with Image.open(file_path) as img:
                img.verify()
            return {
                "path": file_path,
                "type": "image",
                "processor": "image_ocr",
                "notes": "Valid image"
            }
        except Exception:
            return {
                "path": file_path,
                "type": "unknown",
                "processor": "unknown",
                "notes": "Invalid or unreadable image"
            }

    # -------------------------------
    # TEXT FILES
    # -------------------------------
    if ext == ".txt":
        return {
            "path": file_path,
            "type": "text",
            "processor": "text_file",
            "notes": "Plain text file"
        }

    # -------------------------------
    # EMAIL FILES
    # -------------------------------
    if ext in [".eml", ".msg"]:
        return {
            "path": file_path,
            "type": "email",
            "processor": "email_ingest",
            "notes": "Email file"
        }

    # -------------------------------
    # UNKNOWN FILE
    # -------------------------------
    return {
        "path": file_path,
        "type": "unknown",
        "processor": "unknown",
        "notes": "Could not determine file type"
    }


