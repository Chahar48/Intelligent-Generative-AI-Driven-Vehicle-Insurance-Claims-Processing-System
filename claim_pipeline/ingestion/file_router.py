# claim_pipeline/data_ingestion/file_router.py

import os
import fitz  # PyMuPDF
from PIL import Image
import mimetypes
import logging
from dataclasses import dataclass

# ------------------------------------------------------
# Logging Setup
# ------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ------------------------------------------------------
# Data Structure for Routing Result
# ------------------------------------------------------
@dataclass
class FileRoute:
    path: str
    type: str
    recommended_processor: str
    confidence: float = 1.0        # default high confidence
    notes: str = ""                # explanation for routing


# ------------------------------------------------------
# Helper Functions
# ------------------------------------------------------
def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def pdf_has_text_layer(pdf_path: str, sample_pages: int = 2) -> bool:
    """
    Checks first few pages for text content.
    """
    try:
        doc = fitz.open(pdf_path)
        for i in range(min(sample_pages, len(doc))):
            try:
                page = doc.load_page(i)
                text = page.get_text("text")
                if text.strip():
                    return True
            except Exception:
                # Some pages might be corrupt, skip them
                continue
        return False
    except fitz.fitz.FileDataError:
        logger.warning(f"Corrupt or unreadable PDF: {pdf_path}")
        return False
    except Exception as e:
        logger.error(f"Error checking PDF text layer: {pdf_path} | {e}")
        return False


def pdf_contains_email_text(pdf_path: str) -> bool:
    """
    Rare case: some brokers convert emails into PDFs.
    Detect common email headers inside the PDF text.
    """
    try:
        doc = fitz.open(pdf_path)
        max_pages = min(2, len(doc))

        email_indicators = ["From:", "Sent:", "To:", "Subject:", "Cc:", "Forwarded message"]

        for i in range(max_pages):
            try:
                page = doc.load_page(i)
                text = page.get_text("text").lower()

                if any(ind.lower() in text for ind in email_indicators):
                    return True
            except Exception:
                continue

        return False

    except Exception:
        return False


# ------------------------------------------------------
# Main Routing Function
# ------------------------------------------------------
def detect_file_type(file_path: str) -> FileRoute:
    """
    Detect file type and recommend processing method.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    logger.info(f"Routing file: {file_path} (ext: {ext})")

    # ---------------------------
    # PDF
    # ---------------------------
    if ext == ".pdf":
        # Check for email-like content inside PDF
        if pdf_contains_email_text(file_path):
            logger.info(f"Detected embedded email content in PDF: {file_path}")
            return FileRoute(
                path=file_path,
                type="pdf_email",
                recommended_processor="pdf_text",
                confidence=0.7,
                notes="Contains email-like patterns inside PDF"
            )

        # Check for text layer
        has_text = pdf_has_text_layer(file_path)
        processor = "pdf_text" if has_text else "pdf_scanned"

        logger.info(f"PDF routed as: {processor} (Text layer: {has_text})")

        return FileRoute(
            path=file_path,
            type="pdf",
            recommended_processor=processor,
            confidence=0.9 if has_text else 0.7,
            notes="PDF with text layer" if has_text else "Scanned PDF (no text layer detected)"
        )

    # ---------------------------
    # IMAGE
    # ---------------------------
    if ext in [".png", ".jpg", ".jpeg"]:
        try:
            with Image.open(file_path) as img:
                img.verify()
            logger.info(f"Image validated: {file_path}")
            return FileRoute(
                path=file_path,
                type="image",
                recommended_processor="image_ocr",
                confidence=0.95,
                notes="Valid image file"
            )
        except Exception:
            logger.warning(f"Invalid/corrupt image: {file_path}")

    # ---------------------------
    # TEXT
    # ---------------------------
    if ext == ".txt":
        logger.info(f"Text file detected: {file_path}")
        return FileRoute(
            path=file_path,
            type="text",
            recommended_processor="text_file",
            confidence=1.0,
            notes="Plain text file"
        )

    # ---------------------------
    # EMAIL FILES (.eml, .msg)
    # ---------------------------
    if ext in [".eml", ".msg"]:
        logger.info(f"Email file detected: {file_path}")
        return FileRoute(
            path=file_path,
            type="email",
            recommended_processor="email_ingest",
            confidence=1.0,
            notes="Email file format"
        )

    # ---------------------------
    # MIME fallback
    # ---------------------------
    mime_type = mimetypes.guess_type(file_path)[0]
    if mime_type and "text" in mime_type:
        logger.info(f"MIME-based text detection: {file_path} ({mime_type})")
        return FileRoute(
            path=file_path,
            type="text",
            recommended_processor="text_file",
            confidence=0.6,
            notes=f"Detected from MIME type: {mime_type}"
        )

    # ---------------------------
    # Unknown
    # ---------------------------
    logger.warning(f"Unknown/unhandled file type: {file_path}")
    return FileRoute(
        path=file_path,
        type="unknown",
        recommended_processor="unknown",
        confidence=0.1,
        notes="Could not classify"
    )
