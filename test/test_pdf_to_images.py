# test/test_pdf_to_images.py

import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


# Mock fitz globally so the module under test imports the fake version
fitz_mock = MagicMock()
sys.modules["fitz"] = fitz_mock

from claim_pipeline.preprocessing import pdf_to_images


# -------------------------------------------------------------------
# Helper Classes
# -------------------------------------------------------------------
class FakePixmap:
    """Simulates fitz.Pixmap output for saving."""
    def __init__(self, data=b"IMG"):
        self.data = data

    def save(self, path):
        # simulate saving successfully
        pass

    def tostring(self, *args, **kwargs):
        return self.data


class FakePage:
    """Simulates a fitz page object."""
    def __init__(self, pixmap=None):
        self.pixmap = pixmap or FakePixmap()

    def get_pixmap(self, matrix=None):
        return self.pixmap


class FakeDoc:
    """Simulates a PyMuPDF Document with N pages."""
    def __init__(self, pages):
        self._pages = pages

    def load_page(self, idx):
        return self._pages[idx]

    def __len__(self):
        return len(self._pages)


# -------------------------------------------------------------------
# Test: Unsupported format
# -------------------------------------------------------------------
def test_unsupported_format():
    res = pdf_to_images.pdf_to_images("dummy.pdf", "C001", output_format="gif")
    assert res == []


# -------------------------------------------------------------------
# Test: PDF open failure
# -------------------------------------------------------------------
@patch("fitz.open", side_effect=Exception("PDF open error"))
def test_pdf_open_fail(mock_open):
    res = pdf_to_images.pdf_to_images("bad.pdf", "C001")
    assert res == []


# -------------------------------------------------------------------
# Test: Convert PDF with one page
# -------------------------------------------------------------------
@patch("cv2.imwrite", return_value=True)
@patch("cv2.imread", return_value=np.zeros((1200, 900, 3), dtype=np.uint8))
@patch("fitz.open")
def test_pdf_single_page(mock_open, mock_imread, mock_write):
    fake_doc = FakeDoc([FakePage()])
    mock_open.return_value = fake_doc

    res = pdf_to_images.pdf_to_images("invoice.pdf", "C123")

    assert len(res) == 1
    item = res[0]

    assert item.page_number == 1
    assert item.width == 900
    assert item.height == 1200
    assert item.dpi == 300
    assert isinstance(item.sha256, str)
    assert "raw" in item.debug_variants
    assert "deskewed" in item.debug_variants
    assert "thresholded" in item.debug_variants


# -------------------------------------------------------------------
# Test: Low resolution → warning + confidence drop
# -------------------------------------------------------------------
@patch("cv2.imwrite", return_value=True)
@patch("cv2.imread", return_value=np.zeros((500, 400, 3), dtype=np.uint8))  # VERY LOW RES
@patch("fitz.open")
def test_low_resolution_warning(mock_open, mock_imread, mock_write):
    fake_doc = FakeDoc([FakePage()])
    mock_open.return_value = fake_doc

    res = pdf_to_images.pdf_to_images("tiny.pdf", "C001")

    assert len(res) == 1
    info = res[0]

    assert "Low resolution" in info.warnings[0]
    assert info.confidence < 0.8   # dropped from 0.8 → 0.6


# -------------------------------------------------------------------
# Test: Error during page rendering
# -------------------------------------------------------------------
@patch("fitz.open")
def test_render_page_error(mock_open):
    bad_page = MagicMock()
    bad_page.get_pixmap.side_effect = Exception("render failed")

    mock_open.return_value = FakeDoc([bad_page])

    res = pdf_to_images.pdf_to_images("broken.pdf", "C002")

    # Page skipped → returns empty list
    assert res == []


# -------------------------------------------------------------------
# Test: compute_sha256 handles missing file
# -------------------------------------------------------------------
def test_compute_sha256_missing():
    sha = pdf_to_images.compute_sha256("NOT_EXIST.png")
    assert sha == ""


# -------------------------------------------------------------------
# Test: deskew_image handles non-fatal exceptions
# -------------------------------------------------------------------
def test_deskew_image_safe():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = pdf_to_images.deskew_image(img)
    assert isinstance(out, np.ndarray)


# -------------------------------------------------------------------
# Test: enhance_image handles exceptions safely
# -------------------------------------------------------------------
def test_enhance_image_safe():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = pdf_to_images.enhance_image(img)
    assert isinstance(out, np.ndarray)


# -------------------------------------------------------------------
# Test: correct_orientation rotates landscape images
# -------------------------------------------------------------------
def test_correct_orientation_landscape():
    img = np.zeros((400, 900, 3), dtype=np.uint8)  # width > height
    with patch("cv2.rotate", return_value=np.zeros((900, 400, 3), dtype=np.uint8)) as r:
        out = pdf_to_images.correct_orientation(img)
        assert r.called
