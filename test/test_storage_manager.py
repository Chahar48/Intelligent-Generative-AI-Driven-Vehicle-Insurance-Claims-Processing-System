# test/test_storage_manager.py

import os
import json
import hashlib
import shutil
import pytest
from pathlib import Path

from claim_pipeline.storage import storage_manager


@pytest.fixture(autouse=True)
def temp_storage(monkeypatch, tmp_path):
    """
    Redirect BASE_DIR to a temporary folder for all tests.
    Ensures clean isolation.
    """
    base = tmp_path / "claims"
    base.mkdir()

    monkeypatch.setattr(storage_manager, "BASE_DIR", str(base))

    yield

    # cleanup automatically when tmp_path disappears


# Helper to load JSON easily
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------
# TEST: RAW UPLOAD STORAGE
# -------------------------------------------------------------
def test_save_raw_upload_creates_file_and_metadata():
    claim_id = "C001"
    filename = "document.pdf"
    file_bytes = b"fake pdf content 12345"

    saved_path = storage_manager.save_raw_upload(claim_id, filename, file_bytes)

    assert os.path.exists(saved_path)
    assert saved_path.endswith("document.pdf")

    # Metadata file must exist
    meta_path = saved_path + ".json"
    assert os.path.exists(meta_path)

    meta = load_json(meta_path)
    assert meta["file_name"] == filename
    assert "timestamp" in meta
    assert meta["sha256"] == hashlib.sha256(file_bytes).hexdigest()


def test_save_raw_upload_multiple_files():
    claim_id = "C_MULTI"
    fb1 = b"file-1"
    fb2 = b"file-2"

    p1 = storage_manager.save_raw_upload(claim_id, "f1.bin", fb1)
    p2 = storage_manager.save_raw_upload(claim_id, "f2.bin", fb2)

    assert os.path.exists(p1) and os.path.exists(p2)
    assert os.path.exists(p1 + ".json")
    assert os.path.exists(p2 + ".json")


# -------------------------------------------------------------
# TEST: EXTRACTED TEXT STORAGE
# -------------------------------------------------------------
def test_save_extracted_text():
    claim_id = "C002"
    extracted = {"text": "Sample extracted OCR text"}

    path = storage_manager.save_extracted_text(claim_id, extracted)
    assert os.path.exists(path)

    content = load_json(path)
    assert "timestamp" in content
    assert content["extracted_text"] == extracted


# -------------------------------------------------------------
# TEST: STRUCTURED FIELDS STORAGE
# -------------------------------------------------------------
def test_save_structured_fields_all_components():
    claim_id = "C003"
    raw_fields = {"name": "John Doe"}
    normalized = {"name": {"value": "John Doe", "score": 1.0}}
    validation = {"issues": [], "missing_required": []}
    ai_inferred = {"name": {"value": "John Doe", "confidence": 0.95}}

    path = storage_manager.save_structured_fields(
        claim_id,
        raw_fields,
        normalized,
        validation,
        ai_inferred,
    )

    assert os.path.exists(path)

    data = load_json(path)
    assert data["raw_fields"] == raw_fields
    assert data["normalized_fields"] == normalized
    assert data["validation"] == validation
    assert data["ai_inferred_fields"] == ai_inferred
    assert "timestamp" in data


def test_save_structured_fields_without_ai():
    claim_id = "C004"
    raw_fields = {"amount": "10000"}
    normalized = {"amount": {"value": 10000}}
    validation = {"missing_required": ["policy_no"]}

    path = storage_manager.save_structured_fields(
        claim_id, raw_fields, normalized, validation
    )

    data = load_json(path)
    assert data["ai_inferred_fields"] == {}  # empty default


# -------------------------------------------------------------
# TEST: DECISION STORAGE
# -------------------------------------------------------------
def test_save_decision_result():
    claim_id = "C005"
    decision = {"decision": "approve", "confidence": 0.92}

    path = storage_manager.save_decision_result(claim_id, decision)

    assert os.path.exists(path)

    data = load_json(path)
    assert data["decision"] == decision
    assert "timestamp" in data


# -------------------------------------------------------------
# TEST: LOADING FULL CLAIM
# -------------------------------------------------------------
def test_load_claim_reconstructs_all_parts():
    claim_id = "C006"

    # Create raw
    storage_manager.save_raw_upload(claim_id, "file.pdf", b"PDF123")

    # Extracted
    ex = {"text": "abc"}
    storage_manager.save_extracted_text(claim_id, ex)

    # Fields
    raw_fields = {"name": "Alice"}
    normalized = {"name": {"value": "Alice"}}
    validation = {}
    storage_manager.save_structured_fields(
        claim_id, raw_fields, normalized, validation
    )

    # Decision
    decision = {"decision": "review"}
    storage_manager.save_decision_result(claim_id, decision)

    # Now load
    loaded = storage_manager.load_claim(claim_id)

    assert "raw" in loaded
    assert "decision" in loaded
    assert loaded["fields"]["raw_fields"]["name"] == "Alice"
    assert loaded["decision"]["decision"]["decision"] == "review"


# -------------------------------------------------------------
# TEST: LOAD CLAIM MISSING PARTS (should not error)
# -------------------------------------------------------------
def test_load_claim_missing_parts_ok():
    claim_id = "C007"

    # only save raw file
    storage_manager.save_raw_upload(claim_id, "doc.txt", b"abc")

    loaded = storage_manager.load_claim(claim_id)
    # missing fields/extracted/decision should return {}
    assert loaded["extracted"] == {}
    assert loaded["fields"] == {}
    assert loaded["decision"] == {}


# -------------------------------------------------------------
# TEST: ATOMIC JSON WRITE
# -------------------------------------------------------------
def test_json_atomic_write(tmp_path, monkeypatch):
    """
    Ensure .tmp → final replace works properly.
    """
    base = tmp_path / "claims_atomic"
    base.mkdir()
    monkeypatch.setattr(storage_manager, "BASE_DIR", str(base))

    claim_id = "C_ATOMIC"
    path = storage_manager.save_extracted_text(claim_id, {"text": "hello"})

    # There should be no leftover .tmp file
    tmp_version = str(path) + ".tmp"
    assert not os.path.exists(tmp_version)
    assert os.path.exists(path)


# -------------------------------------------------------------
# TEST: HASH FUNCTION
# -------------------------------------------------------------
def test_hash_function_correct():
    data = b"hello world"
    h = storage_manager._hash_bytes(data)
    assert h == hashlib.sha256(data).hexdigest()
