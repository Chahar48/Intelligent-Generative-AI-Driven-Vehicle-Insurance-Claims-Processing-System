import os
import shutil
import time
import json
import datetime
import pytest

import claim_pipeline.security.retention_manager as RM

CLAIMS_DIR = RM.CLAIMS_DIR
ARCHIVE_DIR = RM.ARCHIVE_DIR
MANIFEST_PATH = RM.MANIFEST_PATH


# -----------------------
# Test Helpers
# -----------------------
def _touch_folder(path, days_old: int):
    """Create folder and adjust modified time."""
    os.makedirs(path, exist_ok=True)
    old_time = time.time() - (days_old * 86400)
    os.utime(path, (old_time, old_time))


def setup_module(module):
    """Clean all dirs before test suite."""
    shutil.rmtree(CLAIMS_DIR, ignore_errors=True)
    shutil.rmtree(ARCHIVE_DIR, ignore_errors=True)
    os.makedirs(CLAIMS_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    if os.path.exists(MANIFEST_PATH):
        os.remove(MANIFEST_PATH)


def teardown_module(module):
    """Cleanup after all tests."""
    shutil.rmtree(CLAIMS_DIR, ignore_errors=True)
    shutil.rmtree(ARCHIVE_DIR, ignore_errors=True)
    if os.path.exists(MANIFEST_PATH):
        os.remove(MANIFEST_PATH)


# ============================================================
#                   TEST: ARCHIVE OLD CLAIMS
# ============================================================
def test_archive_old_claims_moves_old_folders():
    old_claim = os.path.join(CLAIMS_DIR, "CLM_OLD")
    _touch_folder(old_claim, days_old=400)  # >1 year

    manifest = RM.archive_old_claims(retention_days=90)

    assert "CLM_OLD" in [x["claim"] for x in manifest["archived"]]
    assert not os.path.exists(old_claim)
    assert os.path.exists(os.path.join(ARCHIVE_DIR, "CLM_OLD"))


def test_archive_old_claims_ignores_recent():
    recent_claim = os.path.join(CLAIMS_DIR, "CLM_NEW")
    _touch_folder(recent_claim, days_old=1)

    manifest = RM.archive_old_claims(retention_days=90)

    # recent should NOT be archived
    assert "CLM_NEW" not in [x["claim"] for x in manifest["archived"]]
    assert os.path.exists(recent_claim)


def test_archive_old_claims_manifest_appends():
    # fresh manifest
    if os.path.exists(MANIFEST_PATH):
        os.remove(MANIFEST_PATH)

    claim_x = os.path.join(CLAIMS_DIR, "CLM_X")
    _touch_folder(claim_x, days_old=200)

    manifest = RM.archive_old_claims(retention_days=50)

    # manifest created
    assert os.path.exists(MANIFEST_PATH)

    data = json.load(open(MANIFEST_PATH))
    assert "archived" in data
    assert any(x["claim"] == "CLM_X" for x in data["archived"])


def test_archive_does_not_overwrite_existing_archive():
    """If archive folder exists, retention manager must rename target."""
    c1 = os.path.join(CLAIMS_DIR, "CLM_DUP")
    c2 = os.path.join(CLAIMS_DIR, "CLM_DUP2")

    # create archived folder with same name
    arch_existing = os.path.join(ARCHIVE_DIR, "CLM_DUP")
    os.makedirs(arch_existing, exist_ok=True)

    # now create two claims older than retention
    _touch_folder(c1, days_old=500)
    _touch_folder(c2, days_old=500)

    manifest = RM.archive_old_claims(retention_days=10)

    # one should use CLM_DUP, the next should create a renamed version
    archived_paths = [entry["moved_to"] for entry in manifest["archived"]]

    assert any("CLM_DUP" in p for p in archived_paths)
    assert any("CLM_DUP_" in p for p in archived_paths)


# ============================================================
#              TEST: DELETE OLD ARCHIVED CLAIMS
# ============================================================
def test_delete_archived_claims_older_than():
    old_arch = os.path.join(ARCHIVE_DIR, "ARCH_OLD")
    _touch_folder(old_arch, days_old=800)

    deleted = RM.delete_archived_claims_older_than(days=365)

    assert any(x["claim"] == "ARCH_OLD" for x in deleted)
    assert not os.path.exists(old_arch)


def test_delete_archived_claims_keeps_recent():
    recent_arch = os.path.join(ARCHIVE_DIR, "ARCH_NEW")
    _touch_folder(recent_arch, days_old=5)

    deleted = RM.delete_archived_claims_older_than(days=365)

    assert not any(x["claim"] == "ARCH_NEW" for x in deleted)
    assert os.path.exists(recent_arch)
