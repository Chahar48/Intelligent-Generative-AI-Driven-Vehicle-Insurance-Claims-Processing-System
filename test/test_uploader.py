# test/test_uploader.py

import sys
import os
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

# ----------------------------------------------------------
# Fake config module so RAW_DATA_DIR import works
# ----------------------------------------------------------
fake_config = MagicMock()
fake_config.RAW_DATA_DIR = "/tmp"   # logical root for tests
sys.modules["config"] = fake_config
sys.modules["config.__init__"] = fake_config

# ----------------------------------------------------------
# Import module under test
# ----------------------------------------------------------
from claim_pipeline.ingestion.uploader import (
    create_claim_folder,
    compute_sha256,
    save_uploaded_files,
    UploadedFileInfo,
)


# ----------------------------------------------------------
# Helper: Fake uploaded file objects
# ----------------------------------------------------------
class FakeFile:
    """Simulates a Streamlit UploadedFile-like object"""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self):
        return self._data

    def read(self):
        return self._data


# ----------------------------------------------------------
# Test: create_claim_folder (cross-platform)
# ----------------------------------------------------------
@patch("os.makedirs")
def test_create_claim_folder(mock_makedirs):
    folder = create_claim_folder("1234")

    # Normalize the path for assertion
    expected_path = os.path.normpath(folder)

    # create_claim_folder returns the correct path
    assert os.path.normpath(folder) == expected_path

    # Verify makedirs was called with EXACTLY the same argument as function
    mock_makedirs.assert_called_once_with(folder, exist_ok=True)


# ----------------------------------------------------------
# Test: compute_sha256 success
# ----------------------------------------------------------
def test_compute_sha256(tmp_path):
    f = tmp_path / "test.bin"
    data = b"hello world"
    f.write_bytes(data)

    result = compute_sha256(str(f))

    import hashlib
    expected = hashlib.sha256(data).hexdigest()

    assert result == expected


# ----------------------------------------------------------
# Test: compute_sha256 missing file
# ----------------------------------------------------------
def test_compute_sha256_missing_file():
    result = compute_sha256("/tmp/does_not_exist.bin")
    assert result is None


# ----------------------------------------------------------
# Test: save_uploaded_files empty list
# ----------------------------------------------------------
def test_save_uploaded_files_empty():
    result = save_uploaded_files([])
    assert result == []


# ----------------------------------------------------------
# Test: save_uploaded_files single file (auto claim ID)
# ----------------------------------------------------------
@patch("claim_pipeline.ingestion.uploader.create_claim_folder")
def test_save_uploaded_files_single(mock_create_folder, tmp_path):
    mock_create_folder.return_value = str(tmp_path)

    fake_file = FakeFile("doc.pdf", b"PDFDATA")

    results = save_uploaded_files([fake_file])

    assert len(results) == 1
    info = results[0]

    assert isinstance(info, UploadedFileInfo)
    assert info.saved is True
    assert info.size == len(b"PDFDATA")
    assert info.sha256 is not None
    assert info.confidence == 0.95


# ----------------------------------------------------------
# Test: save_uploaded_files with custom claim_id
# ----------------------------------------------------------
@patch("claim_pipeline.ingestion.uploader.create_claim_folder")
def test_save_uploaded_files_custom_claim_id(mock_create_folder, tmp_path):
    mock_create_folder.return_value = str(tmp_path)

    fake_file = FakeFile("image.png", b"IMAGEDATA")

    results = save_uploaded_files([fake_file], claim_id="C123")

    assert results[0].claim_id == "C123"


# ----------------------------------------------------------
# Test: save_uploaded_files multiple files
# ----------------------------------------------------------
@patch("claim_pipeline.ingestion.uploader.create_claim_folder")
def test_save_uploaded_files_multiple(mock_create_folder, tmp_path):
    mock_create_folder.return_value = str(tmp_path)

    files = [
        FakeFile("a.txt", b"A"),
        FakeFile("b.txt", b"B"),
        FakeFile("c.txt", b"C")
    ]

    results = save_uploaded_files(files, claim_id="C001")

    assert len(results) == 3
    assert {r.original_name for r in results} == {"a.txt", "b.txt", "c.txt"}


# ----------------------------------------------------------
# Test: save_uploaded_files write failure
# ----------------------------------------------------------
@patch("builtins.open", side_effect=Exception("disk full"))
@patch("claim_pipeline.ingestion.uploader.create_claim_folder")
def test_save_uploaded_files_write_failure(mock_create_folder, mock_open, tmp_path):
    mock_create_folder.return_value = str(tmp_path)

    fake_file = FakeFile("fail.txt", b"FAILDATA")

    results = save_uploaded_files([fake_file], claim_id="FAIL")

    info = results[0]

    assert info.saved is False
    assert info.sha256 is None
    assert "disk full" in info.notes
