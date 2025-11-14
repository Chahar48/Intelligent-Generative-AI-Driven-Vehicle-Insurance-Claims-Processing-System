# claim_pipeline/storage/storage_manager.py
"""
Unified Storage Manager for the Claims Pipeline.

Stores:
    ✓ Raw uploaded files
    ✓ Extracted text (PDF/OCR/Email)
    ✓ Structured fields (regex + AI + normalized)
    ✓ Validation output
    ✓ Final decisions (rule + AI)
    
Storage Format (JSON folders):
data/
 └── claims/
        <claim_id>/
            raw/
            extracted/
            fields/
            decision/

This file is built to support auditability, deterministic reconstruction,
and future upgrade to PostgreSQL without changing pipeline logic.
"""

from __future__ import annotations
import os
import json
import hashlib
import datetime
from typing import Dict, Any

# ---------------------
# Base folder
# ---------------------
BASE_DIR = "data/claims"


# ---------------------
# Utilities
# ---------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return datetime.datetime.utcnow().isoformat()


def _write_json(path: str, content: Dict[str, Any]):
    """Safely write JSON to disk."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------
# 1️⃣ RAW UPLOAD STORAGE
# ---------------------------------------------------
def save_raw_upload(claim_id: str, file_name: str, file_bytes: bytes) -> str:
    """
    Saves raw uploaded file bytes.
    Returns path to saved file and stores audit metadata.
    """
    claim_path = os.path.join(BASE_DIR, claim_id, "raw")
    _ensure_dir(claim_path)

    file_path = os.path.join(claim_path, file_name)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    hash_hex = _hash_bytes(file_bytes)

    metadata = {
        "file_name": file_name,
        "saved_path": file_path,
        "sha256": hash_hex,
        "timestamp": _timestamp()
    }

    meta_path = os.path.join(claim_path, f"{file_name}.json")
    _write_json(meta_path, metadata)

    return file_path


# ---------------------------------------------------
# 2️⃣ EXTRACTED TEXT STORAGE
# ---------------------------------------------------
def save_extracted_text(claim_id: str, extracted: Dict[str, Any]) -> str:
    """
    Stores extracted text from PDF/OCR/email.
    """
    path = os.path.join(BASE_DIR, claim_id, "extracted")
    _ensure_dir(path)

    out = {
        "timestamp": _timestamp(),
        "extracted_text": extracted
    }

    save_path = os.path.join(path, "text.json")
    _write_json(save_path, out)

    return save_path


# ---------------------------------------------------
# 3️⃣ STRUCTURED FIELDS STORAGE
# ---------------------------------------------------
def save_structured_fields(
    claim_id: str,
    raw_fields: Dict[str, Any],
    normalized_fields: Dict[str, Any],
    validation: Dict[str, Any],
    ai_inferred: Dict[str, Any] = None
) -> str:
    """
    Stores:
    - raw extracted fields (regex)
    - normalized fields
    - validation output
    - AI-inferred fields (optional)
    """
    path = os.path.join(BASE_DIR, claim_id, "fields")
    _ensure_dir(path)

    out = {
        "timestamp": _timestamp(),
        "raw_fields": raw_fields,
        "normalized_fields": normalized_fields,
        "validation": validation,
        "ai_inferred_fields": ai_inferred or {}
    }

    save_path = os.path.join(path, "fields.json")
    _write_json(save_path, out)

    return save_path


# ---------------------------------------------------
# 4️⃣ DECISION RESULT STORAGE
# ---------------------------------------------------
def save_decision_result(claim_id: str, decision: Dict[str, Any]) -> str:
    """
    Stores final decision (approve/review/reject),
    flags, evidence, confidence, and ai reasoning.
    """
    path = os.path.join(BASE_DIR, claim_id, "decision")
    _ensure_dir(path)

    out = {
        "timestamp": _timestamp(),
        "decision": decision
    }

    save_path = os.path.join(path, "decision.json")
    _write_json(save_path, out)

    return save_path


# ---------------------------------------------------
# LOADER (Optional)
# ---------------------------------------------------
def load_claim(claim_id: str) -> Dict[str, Any]:
    """
    Load all stored data for a claim for reconstruction.
    """
    base = os.path.join(BASE_DIR, claim_id)

    def load_json(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None

    return {
        "raw": load_json(os.path.join(base, "raw", f"{claim_id}.json")) or {},
        "extracted": load_json(os.path.join(base, "extracted", "text.json")) or {},
        "fields": load_json(os.path.join(base, "fields", "fields.json")) or {},
        "decision": load_json(os.path.join(base, "decision", "decision.json")) or {}
    }


class StorageManager:
    def save_raw_upload(self, claim_id: str, file_path: str, uploader: str):
        # file_path is a local file, not bytes. So read it.
        file_name = os.path.basename(file_path)

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        return save_raw_upload(claim_id, file_name, file_bytes)

    def save_extracted_text(self, claim_id: str, extracted_text: str):
        return save_extracted_text(claim_id, extracted_text)

    def save_structured_fields(self, claim_id, fields_dict: dict):
        return save_structured_fields(
            claim_id,
            raw_fields=fields_dict.get("raw_fields", {}),
            normalized_fields=fields_dict.get("normalized", {}),
            validation=fields_dict.get("validation", {}),
            ai_inferred=fields_dict.get("ai_inference", {})
        )

    def save_decision(self, claim_id, decision):
        return save_decision_result(claim_id, decision)

    def save_hitl_task(self, claim_id, task):
        # optional if you store HITL tasks later
        path = os.path.join(BASE_DIR, claim_id, "hitl")
        _ensure_dir(path)
        out = {"timestamp": _timestamp(), "task": task}
        save_path = os.path.join(path, "hitl.json")
        _write_json(save_path, out)
        return save_path
