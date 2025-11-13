# test/test_pdf_extractor.py

import sys
import pytest
from unittest.mock import MagicMock, patch

# Patch PyMuPDF (fitz) module to avoid real dependency
fitz_mock = MagicMock()
sys.modules["fitz"] = fitz_mock

from claim_pipeline.extraction import pdf_extractor


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
class FakePage:
    """Simulates a PDF page with optional blocks and pixmap bytes."""
    def __init__(self, blocks=None, pixmap_bytes=b"IMG"):
        self.blocks = blocks or []
        self.pixmap_bytes = pixmap_bytes

    def get_text(self, mode):
        if mode == "blocks":
            return self.blocks
        return ""

    def get_pixmap(self):
        pix = MagicMock()
        pix.tobytes.return_value = self.pixmap_bytes
        return pix


class FakeDoc:
    """Simulates a fitz document."""
    def __init__(self, pages):
        self._pages = pages

    def load_page(self, idx):
        return self._pages[idx]

    def __len__(self):
        return len(self._pages)


# -------------------------------------------------------------------
# Test: SHA256 calculation for PDF
# -------------------------------------------------------------------
def test_compute_pdf_sha(tmp_path):
    pdf = tmp_path / "file.pdf"
    pdf.write_bytes(b"hello-pdf")

    sha = pdf_extractor._compute_pdf_sha(str(pdf))
    import hashlib
    expected = hashlib.sha256(b"hello-pdf").hexdigest()

    assert sha == expected


def test_compute_pdf_sha_missing_file():
    sha = pdf_extractor._compute_pdf_sha("missing.pdf")
    assert sha == ""


# -------------------------------------------------------------------
# Test: Heuristic table detection (true & false)
# -------------------------------------------------------------------
def test_heuristic_table_detect_true():
    # Create 3 columns each repeated >=2 times so repeated >= 3 as the function requires
    blocks = [
        (10, 10, 50, 20, "A", 1),
        (10, 30, 50, 40, "B", 2),
        (10, 50, 50, 60, "C", 3),
        (100, 10, 140, 20, "D", 4),
        (100, 30, 140, 40, "E", 5),
        (100, 50, 140, 60, "F", 6),
        (200, 10, 240, 20, "G", 7),
        (200, 30, 240, 40, "H", 8),
        (200, 50, 240, 60, "I", 9),
    ]
    assert pdf_extractor._heuristic_table_detect(blocks) is True


def test_heuristic_table_detect_false():
    blocks = [(10, 10, 50, 20, "A", 1)]
    assert pdf_extractor._heuristic_table_detect(blocks) is False


# -------------------------------------------------------------------
# Test: Page confidence heuristic
# -------------------------------------------------------------------
def test_page_confidence():
    conf = pdf_extractor._page_confidence("some text" * 20, 10, False)
    assert isinstance(conf, float)
    assert 0.05 <= conf <= 0.99


# -------------------------------------------------------------------
# Test: extract_text_from_pdf → valid single page
# -------------------------------------------------------------------
@patch("fitz.open")
def test_extract_text_single_page(mock_open):
    page = FakePage(
        blocks=[
            (10, 10, 50, 20, "Hello", 1),
            (10, 30, 50, 40, "World", 2),
        ]
    )
    mock_open.return_value = FakeDoc([page])

    pdf_path = "dummy.pdf"
    res = pdf_extractor.extract_text_from_pdf(pdf_path)

    assert res["num_pages"] == 1
    assert "Hello" in res["text"]
    assert "World" in res["text"]
    assert res["pages"][0]["page_no"] == 1
    assert res["pages"][0]["table_detected"] is False


# -------------------------------------------------------------------
# Test: extract_text_from_pdf → empty page (fallback disabled)
# -------------------------------------------------------------------
@patch("fitz.open")
def test_extract_text_empty_page_no_fallback(mock_open):
    page = FakePage(blocks=[])
    mock_open.return_value = FakeDoc([page])

    res = pdf_extractor.extract_text_from_pdf("empty.pdf", fallback_to_ocr=False)

    assert res["text"] == ""
    assert res["fallback_used"] is False

    # The code may append either "No selectable text; hashed page image." first
    # and then "Page X has no extracted text." — accept either as evidence of empty page handling.
    notes = res["pages"][0]["notes"]
    joined = " ".join(n.lower() for n in notes)
    assert ("no selectable text" in joined) or ("no extracted text" in joined)


# -------------------------------------------------------------------
# Test: extract_text_from_pdf → fallback_to_ocr = True
# (create attribute if not present using create=True)
# -------------------------------------------------------------------
@patch("fitz.open")
@patch("claim_pipeline.extraction.ocr_extractor.ocr_pdf_scanned", return_value={"text": "OCR extracted"}, create=True)
def test_extract_text_pdf_fallback(mock_ocr, mock_open):
    # page with NO selectable text
    page = FakePage(blocks=[])
    mock_open.return_value = FakeDoc([page])

    res = pdf_extractor.extract_text_from_pdf("scan.pdf", fallback_to_ocr=True)

    assert res["fallback_used"] is True
    assert "OCR extracted" in res["text"]


# -------------------------------------------------------------------
# Test: extract_text_from_pdf → per-page get_text failure
# -------------------------------------------------------------------
@patch("fitz.open")
def test_extract_text_page_exception(mock_open):
    # Simulate get_text("blocks") raising an exception (handled internally)
    bad_page = MagicMock()
    bad_page.get_text.side_effect = Exception("page error")
    # Provide get_pixmap so image hash fallback is attempted
    bad_page.get_pixmap.return_value.tobytes.return_value = b"IMG"

    mock_open.return_value = FakeDoc([bad_page])

    res = pdf_extractor.extract_text_from_pdf("broken.pdf")

    # When get_text("blocks") fails but page processed, confidence is computed from empty text -> base 0.2
    assert res["pages"][0]["confidence"] == 0.2
    assert any("error" in n.lower() for n in res["pages"][0]["notes"] + [""]) or "No selectable text" in " ".join(res["pages"][0]["notes"])
    assert res["num_pages"] == 1


# -------------------------------------------------------------------
# Test: extract_text_from_pdf → full PDF reading crash
# -------------------------------------------------------------------
@patch("fitz.open", side_effect=Exception("PDF open error"))
def test_pdf_open_fail(mock_open):
    res = pdf_extractor.extract_text_from_pdf("badfile.pdf")
    assert res["text"] == ""
    assert res["num_pages"] == 0
