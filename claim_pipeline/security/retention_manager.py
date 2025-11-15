"""
Retention Manager – simplified, stable, and audit-friendly.

Handles:
- Archiving old claim folders
- Deleting old archived claim folders
- Maintaining a retention manifest

No change in behavior — only cleaner & safer.
"""

import os
import shutil
import json
import datetime
from typing import Optional, Dict, Any

CLAIMS_DIR = "data/claims"
ARCHIVE_DIR = "data/archive"
MANIFEST_PATH = "data/retention_manifest.json"

DEFAULT_RETENTION_DAYS = 365 * 5   # Keep claims for 5 years
DEFAULT_ARCHIVE_DELETE_DAYS = 365 * 2

os.makedirs(CLAIMS_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)


# -------------------------------
# Helpers
# -------------------------------
def _utcnow() -> datetime.datetime:
    return datetime.datetime.utcnow()


def _folder_mtime(path: str) -> datetime.datetime:
    """Return folder modified time as datetime."""
    try:
        ts = os.path.getmtime(path)
        return datetime.datetime.utcfromtimestamp(ts)
    except Exception:
        return _utcnow()  # safe fallback


def _load_manifest() -> Dict[str, Any]:
    """Load or initialize retention manifest."""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"archived": [], "deleted": []}


def _save_manifest(manifest: Dict[str, Any]):
    """Write manifest safely."""
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# -------------------------------
# ARCHIVING
# -------------------------------
def archive_old_claims(retention_days: Optional[int] = None):
    """
    Move claim folders older than `retention_days` into ARCHIVE_DIR.
    Returns updated manifest.
    """
    retention_days = retention_days or DEFAULT_RETENTION_DAYS
    cutoff = _utcnow() - datetime.timedelta(days=retention_days)

    manifest = _load_manifest()

    for claim_id in os.listdir(CLAIMS_DIR):
        claim_path = os.path.join(CLAIMS_DIR, claim_id)
        if not os.path.isdir(claim_path):
            continue

        if _folder_mtime(claim_path) >= cutoff:
            continue  # still fresh, skip

        # Compute archive destination
        dest = os.path.join(ARCHIVE_DIR, claim_id)
        if os.path.exists(dest):
            # Avoid collision when archiving multiple times
            timestamp = int(os.path.getmtime(claim_path))
            dest = f"{dest}_{timestamp}"

        shutil.move(claim_path, dest)

        manifest["archived"].append({
            "claim": claim_id,
            "moved_to": dest,
            "time": _utcnow().isoformat()
        })

    _save_manifest(manifest)
    return manifest


# -------------------------------
# DELETION
# -------------------------------
def delete_archived_claims_older_than(days: int = DEFAULT_ARCHIVE_DELETE_DAYS):
    """
    Permanently delete archived claims older than `days`.
    Use with caution.
    """
    cutoff = _utcnow() - datetime.timedelta(days=days)
    manifest = _load_manifest()
    deleted_entries = []

    for folder_name in os.listdir(ARCHIVE_DIR):
        folder_path = os.path.join(ARCHIVE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        if _folder_mtime(folder_path) >= cutoff:
            continue  # not old enough

        shutil.rmtree(folder_path)

        deleted_entries.append({
            "claim": folder_name,
            "deleted_at": _utcnow().isoformat()
        })

    if deleted_entries:
        manifest["deleted"].extend(deleted_entries)
        _save_manifest(manifest)

    return deleted_entries



# # claim_pipeline/security/retention_manager.py
# """
# Document retention manager for JSON-based storage.

# - Archives or deletes claim folders older than retention period.
# - Writes a retention manifest for audit.
# """

# import os
# import shutil
# import json
# import datetime
# from typing import Optional

# CLAIMS_DIR = "data/claims"
# ARCHIVE_DIR = "data/archive"
# MANIFEST_PATH = "data/retention_manifest.json"

# # Defaults (days)
# DEFAULT_RETENTION_DAYS = 365 * 5  # 5 years for claims
# TEMP_RETENTION_DAYS = 30          # temp data

# os.makedirs(CLAIMS_DIR, exist_ok=True)
# os.makedirs(ARCHIVE_DIR, exist_ok=True)

# def _now_date():
#     return datetime.datetime.utcnow()

# def _folder_mtime(path: str) -> datetime.datetime:
#     # Use last modified time of the folder (or creation time)
#     ts = os.path.getmtime(path)
#     return datetime.datetime.utcfromtimestamp(ts)

# def _load_manifest() -> dict:
#     if os.path.exists(MANIFEST_PATH):
#         with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return {"archived": [], "deleted": []}

# def _save_manifest(manifest: dict):
#     with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
#         json.dump(manifest, f, indent=2)

# def archive_old_claims(retention_days: Optional[int] = None):
#     """
#     Move claim folders older than retention_days into ARCHIVE_DIR.
#     """
#     retention_days = retention_days if retention_days is not None else DEFAULT_RETENTION_DAYS
#     cutoff = _now_date() - datetime.timedelta(days=retention_days)
#     manifest = _load_manifest()

#     for folder in os.listdir(CLAIMS_DIR):
#         claim_path = os.path.join(CLAIMS_DIR, folder)
#         if not os.path.isdir(claim_path):
#             continue
#         mtime = _folder_mtime(claim_path)
#         if mtime < cutoff:
#             dest = os.path.join(ARCHIVE_DIR, folder)
#             if os.path.exists(dest):
#                 # avoid overwrite by renaming
#                 dest = dest + "_" + str(int(os.path.getmtime(claim_path)))
#             shutil.move(claim_path, dest)
#             manifest["archived"].append({
#                 "claim": folder,
#                 "moved_to": dest,
#                 "time": datetime.datetime.utcnow().isoformat()
#             })
#     _save_manifest(manifest)
#     return manifest

# def delete_archived_claims_older_than(days: int = 365 * 2):
#     """
#     Permanently delete archived claims older than X days (use with caution).
#     """
#     cutoff = _now_date() - datetime.timedelta(days=days)
#     manifest = _load_manifest()
#     deleted = []
#     for name in os.listdir(ARCHIVE_DIR):
#         path = os.path.join(ARCHIVE_DIR, name)
#         if not os.path.isdir(path):
#             continue
#         mtime = _folder_mtime(path)
#         if mtime < cutoff:
#             shutil.rmtree(path)
#             deleted.append({"claim": name, "deleted_at": _now_date().isoformat()})
#     manifest["deleted"].extend(deleted)
#     _save_manifest(manifest)
#     return deleted
