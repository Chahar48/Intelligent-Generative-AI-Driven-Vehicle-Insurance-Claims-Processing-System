# test/test_ocr_extractor.py

import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# ----------------------------------------------------------
# PATCH cv2 + paddleocr + pytesseract globally
# ----------------------------------------------------------
sys.modules["paddleocr"] = MagicMock()
sys.modules["pytesseract"] = MagicMock()

import cv2
import pytesseract
from claim_pipeline.extraction import ocr_extractor


# ==========================================================
# Test compute_sha256_bytes
# ==========================================================
def test_compute_sha256_bytes():
    data = b"hello"
    result = ocr_extractor.compute_sha256_bytes(data)

    import hashlib
    expected = hashlib.sha256(data).hexdigest()
    assert result == expected


# ==========================================================
# Test compute_sha256_file
# ==========================================================
def test_compute_sha256_file(tmp_path):
    file = tmp_path / "a.bin"
    data = b"123abc"
    file.write_bytes(data)

    result = ocr_extractor.compute_sha256_file(str(file))

    import hashlib
    expected = hashlib.sha256(data).hexdigest()
    assert result == expected


# ==========================================================
# Test load_image_cv2 (mock cv2.imread)
# ==========================================================
@patch("cv2.imread")
def test_load_image_cv2(mock_imread):
    mock_imread.return_value = np.zeros((100, 200, 3), dtype=np.uint8)

    img = ocr_extractor.load_image_cv2("dummy")

    assert img.shape == (100, 200, 3)
    mock_imread.assert_called_once()


# ==========================================================
# Test detect_blur
# ==========================================================
def test_detect_blur():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = ocr_extractor.detect_blur(img)

    # Accept numpy boolean OR python boolean
    assert isinstance(result, (bool, np.bool_))


# ==========================================================
# Test _select_best_engine
# ==========================================================
def test_select_best_engine():
    paddle = {"confidence": 0.9, "text": "abc"}
    tesseract = {"confidence": 0.8, "text": "xyz"}

    chosen_engine, conf, reason = ocr_extractor._select_best_engine(paddle, tesseract)

    assert chosen_engine == "paddle"
    assert conf == 0.9
    assert "paddle" in reason


# ==========================================================
# MAIN END-TO-END TEST: ocr_image_best
# ==========================================================
@patch("cv2.imread")
@patch("claim_pipeline.extraction.ocr_extractor.enhance_variants")
@patch("claim_pipeline.extraction.ocr_extractor._tesseract_engine")
@patch("claim_pipeline.extraction.ocr_extractor._paddle_ocr_engine")
@patch("cv2.imwrite")
def test_ocr_image_best(
    mock_imwrite,
    mock_paddle_engine,
    mock_tess_engine,
    mock_variants,
    mock_imread,
    tmp_path,
):

    # ------------------------------------------------------
    # 1. Mock load image
    # ------------------------------------------------------
    dummy_img = np.zeros((1200, 800, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_img

    # ------------------------------------------------------
    # 2. Mock enhancement variants
    # ------------------------------------------------------
    mock_variants.return_value = {
        "adaptive": np.zeros((100, 100), dtype=np.uint8)
    }

    # ------------------------------------------------------
    # 3. Mock PaddleOCR engine output
    # ------------------------------------------------------
    mock_paddle_engine.return_value = (
        "paddle text",
        0.8,
        [{"text": "A", "bbox": [0, 0, 10, 10], "conf": 80}],
    )

    # ------------------------------------------------------
    # 4. Mock Tesseract output
    # ------------------------------------------------------
    mock_tess_engine.return_value = (
        "tess text",
        0.5,
        [{"text": "B", "bbox": [0, 0, 10, 10], "conf": 50}],
    )

    # ------------------------------------------------------
    # 5. Mock actual PaddleOCR class
    # ------------------------------------------------------
    fake_paddle = MagicMock()
    sys.modules["paddleocr"].PaddleOCR = MagicMock(return_value=fake_paddle)
    ocr_extractor.PaddleOCR = MagicMock(return_value=fake_paddle)

    # ------------------------------------------------------
    # Create temp image file for hashing
    # ------------------------------------------------------
    img_path = tmp_path / "img.png"
    img_path.write_bytes(b"\x89PNG....fake....")

    # ------------------------------------------------------
    # Run OCR
    # ------------------------------------------------------
    result = ocr_extractor.ocr_image_best(str(img_path))

    # ------------------------------------------------------
    # Assertions
    # ------------------------------------------------------
    assert result["engine"] == "paddle"            # paddle higher confidence
    assert result["confidence"] <= 1.0             # safe numeric
    assert "paddle text" in result["text"]         # paddle output chosen
    assert "tess text" not in result["text"]       # tesseract ignored
    assert "lines" in result
    assert isinstance(result["lines"], list)

    # Metadata checks
    assert "width" in result["metadata"]
    assert "height" in result["metadata"]
    assert "runtime_sec" in result["metadata"]


# ==========================================================
# Test fallback when both engines return nothing
# ==========================================================
@patch("cv2.imread")
@patch("claim_pipeline.extraction.ocr_extractor.enhance_variants")
@patch("claim_pipeline.extraction.ocr_extractor._tesseract_engine")
@patch("claim_pipeline.extraction.ocr_extractor._paddle_ocr_engine")
def test_ocr_no_output(
    mock_paddle_engine,
    mock_tess_engine,
    mock_variants,
    mock_imread,
    tmp_path,
):

    mock_imread.return_value = np.zeros((1000, 800, 3), dtype=np.uint8)

    mock_variants.return_value = {
        "adaptive": np.zeros((100, 100), dtype=np.uint8)
    }

    mock_paddle_engine.return_value = ("", 0.0, [])
    mock_tess_engine.return_value = ("", 0.0, [])

    img_path = tmp_path / "img2.png"
    img_path.write_bytes(b"fake")

    result = ocr_extractor.ocr_image_best(str(img_path))

    assert result["engine"] in ["none", None]
    assert result["confidence"] == 0.0
    assert result["text"] == ""
