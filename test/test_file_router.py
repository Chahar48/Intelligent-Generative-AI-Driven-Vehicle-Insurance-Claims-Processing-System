# tests/test_file_router.py

import sys
from unittest.mock import MagicMock

# Mock fitz so test collection does not fail
sys.modules["fitz"] = MagicMock()

import pytest
from unittest.mock import patch, MagicMock
from claim_pipeline.ingestion.file_router import (
    detect_file_type,
    FileRoute,
)

# ----------------------------------------------------------------
# Helper: Create dummy files (empty files are enough for routing)
# ----------------------------------------------------------------
@pytest.fixture
def create_dummy_file(tmp_path):
    def _create(filename):
        file_path = tmp_path / filename
        file_path.write_bytes(b"")   # create empty file
        return str(file_path)
    return _create


# ----------------------------------------------------------------
# Test: PDF with text layer
# ----------------------------------------------------------------
@patch("claim_pipeline.ingestion.file_router.pdf_has_text_layer", return_value=True)
def test_pdf_with_text(mock_pdf_text, create_dummy_file):
    file_path = create_dummy_file("sample.pdf")

    result = detect_file_type(file_path)

    assert isinstance(result, FileRoute)
    assert result.type == "pdf"
    assert result.recommended_processor == "pdf_text"
    assert result.confidence == 0.9


# ----------------------------------------------------------------
# Test: Scanned PDF (no text layer)
# ----------------------------------------------------------------
@patch("claim_pipeline.ingestion.file_router.pdf_has_text_layer", return_value=False)
def test_scanned_pdf(mock_pdf_text, create_dummy_file):
    file_path = create_dummy_file("scan.pdf")

    result = detect_file_type(file_path)

    assert result.type == "pdf"
    assert result.recommended_processor == "pdf_scanned"
    assert result.confidence == 0.7


# ----------------------------------------------------------------
# Test: PDF containing email text
# ----------------------------------------------------------------
@patch("claim_pipeline.ingestion.file_router.pdf_contains_email_text", return_value=True)
def test_pdf_with_email_text(mock_pdf_email, create_dummy_file):
    file_path = create_dummy_file("email_like.pdf")

    result = detect_file_type(file_path)

    assert result.type == "pdf_email"
    assert result.recommended_processor == "pdf_text"
    assert result.confidence == 0.7


# ----------------------------------------------------------------
# Test: Image file
# ----------------------------------------------------------------
@patch("PIL.Image.open")
def test_image_file(mock_image, create_dummy_file):
    file_path = create_dummy_file("image.jpg")

    mock_verify = MagicMock()
    mock_image.return_value.__enter__.return_value.verify = mock_verify

    result = detect_file_type(file_path)

    assert result.type == "image"
    assert result.recommended_processor == "image_ocr"
    assert result.confidence == 0.95


# ----------------------------------------------------------------
# Test: Text file
# ----------------------------------------------------------------
def test_text_file(create_dummy_file):
    file_path = create_dummy_file("note.txt")

    result = detect_file_type(file_path)

    assert result.type == "text"
    assert result.recommended_processor == "text_file"


# ----------------------------------------------------------------
# Test: Email (.eml)
# ----------------------------------------------------------------
def test_email_eml(create_dummy_file):
    file_path = create_dummy_file("mail.eml")

    result = detect_file_type(file_path)

    assert result.type == "email"
    assert result.recommended_processor == "email_ingest"


# ----------------------------------------------------------------
# Test: Email (.msg)
# ----------------------------------------------------------------
def test_email_msg(create_dummy_file):
    file_path = create_dummy_file("mail.msg")

    result = detect_file_type(file_path)

    assert result.type == "email"
    assert result.recommended_processor == "email_ingest"


# ----------------------------------------------------------------
# Test: Unknown extension
# ----------------------------------------------------------------
def test_unknown_file(create_dummy_file):
    file_path = create_dummy_file("unknown.xyz")

    result = detect_file_type(file_path)

    assert result.type == "unknown"
    assert result.recommended_processor == "unknown"
