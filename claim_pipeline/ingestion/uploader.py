# claim_pipeline/data_ingestion/uploader.py

import os
import uuid
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path
from config import RAW_DATA_DIR


# ------------------------------------------------------
# Logging config
# ------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ------------------------------------------------------
# Dataclass for file metadata
# ------------------------------------------------------
@dataclass
class UploadedFileInfo:
    claim_id: str
    original_name: str
    saved_path: str
    size: int
    sha256: str
    saved: bool
    confidence: float = 1.0
    notes: str = ""


# ------------------------------------------------------
# Helper: create claim folder
# ------------------------------------------------------
def create_claim_folder(claim_id: str) -> str:
    """
    Creates a folder for the claim under RAW_DATA_DIR.
    """
    folder = os.path.join(RAW_DATA_DIR, claim_id)
    os.makedirs(folder, exist_ok=True)
    logger.info(f"[Uploader] Claim folder created: {folder}")
    return folder


# ------------------------------------------------------
# Helper: compute sha256 hash of saved file
# ------------------------------------------------------
def compute_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"[Uploader] Failed to compute SHA256: {file_path} | {e}")
        return None


# ------------------------------------------------------
# Main Function: Save uploaded files
# ------------------------------------------------------
def save_uploaded_files(uploaded_files: List, claim_id: str = None) -> List[UploadedFileInfo]:
    """
    Save uploaded files and return a list of UploadedFileInfo.
    Supports Streamlit UploadedFile or any file-like object.
    """
    if not uploaded_files:
        logger.warning("[Uploader] No files received to upload.")
        return []

    # Generate claim ID if not provided
    if claim_id is None:
        claim_id = f"claim_{uuid.uuid4().hex[:8]}"
        logger.info(f"[Uploader] Generated new claim_id: {claim_id}")

    claim_folder = create_claim_folder(claim_id)
    results = []

    for file_obj in uploaded_files:
        filename = Path(getattr(file_obj, "name", "uploaded_file")).name
        safe_filename = filename.replace("..", "").replace("/", "_")
        save_path = os.path.join(claim_folder, safe_filename)

        try:
            # Write bytes to disk
            with open(save_path, "wb") as out:
                if hasattr(file_obj, "getbuffer"):
                    out.write(file_obj.getbuffer())
                else:
                    out.write(file_obj.read())

            size = os.path.getsize(save_path)
            sha256 = compute_sha256(save_path)

            info = UploadedFileInfo(
                claim_id=claim_id,
                original_name=filename,
                saved_path=save_path,
                size=size,
                sha256=sha256,
                saved=True,
                confidence=0.95,
                notes="File saved successfully"
            )

            logger.info(f"[Uploader] Saved file: {filename} → {save_path} ({size} bytes)")

        except Exception as e:
            logger.error(f"[Uploader] FAILED to save file {filename}: {e}")

            info = UploadedFileInfo(
                claim_id=claim_id,
                original_name=filename,
                saved_path=save_path,
                size=0,
                sha256=None,
                saved=False,
                confidence=0.2,
                notes=str(e)
            )

        results.append(info)

    return results
