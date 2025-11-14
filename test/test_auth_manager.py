# test/test_auth_manager.py

import os
import json
import pytest
from pathlib import Path

from claim_pipeline.security import auth_manager


@pytest.fixture(autouse=True)
def temp_user_store(monkeypatch, tmp_path):
    """
    Redirect USERS_PATH to a temporary location for all tests.
    Ensures the real system is not modified.
    """
    users_path = tmp_path / "users.json"
    monkeypatch.setattr(auth_manager, "USERS_PATH", str(users_path))

    # Ensure default users are created for tests
    with open(users_path, "w", encoding="utf-8") as f:
        json.dump(auth_manager.DEFAULT_USERS, f, indent=2)

    yield


def load_users():
    with open(auth_manager.USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------
# TEST: USER LOADING
# ---------------------------------------------------------------------------------------------------
def test_load_users_reads_file():
    users = load_users()
    assert users["admin"] == "admin"
    assert "alice_reviewer" in users


def test_get_user_role_valid():
    assert auth_manager.get_user_role("admin") == "admin"
    assert auth_manager.get_user_role("alice_reviewer") == "reviewer"


def test_get_user_role_missing():
    assert auth_manager.get_user_role("unknown_user") is None


# ---------------------------------------------------------------------------------------------------
# TEST: ROLE CHECKING
# ---------------------------------------------------------------------------------------------------
def test_has_role_exact_match():
    assert auth_manager.has_role("alice_reviewer", "reviewer") is True
    assert auth_manager.has_role("viewer_user", "viewer") is True


def test_has_role_invalid_user():
    assert auth_manager.has_role("not_exist", "reviewer") is False


def test_has_role_admin_super_permissions():
    # Admin must access reviewer, auditor, viewer
    assert auth_manager.has_role("admin", "reviewer") is True
    assert auth_manager.has_role("admin", "auditor") is True
    assert auth_manager.has_role("admin", "viewer") is True


def test_has_role_reviewer_cannot_access_admin():
    assert auth_manager.has_role("alice_reviewer", "admin") is False


# ---------------------------------------------------------------------------------------------------
# TEST: DECORATOR require_role
# ---------------------------------------------------------------------------------------------------
def test_require_role_allows_access():
    @auth_manager.require_role("reviewer")
    def sample(username):
        return f"{username} OK"

    assert sample("alice_reviewer") == "alice_reviewer OK"


def test_require_role_blocks_access():
    @auth_manager.require_role("admin")
    def admin_only(username):
        return "secret"

    with pytest.raises(PermissionError):
        admin_only("viewer_user")


def test_require_role_admin_allowed_everywhere():
    @auth_manager.require_role("auditor")
    def audit(username):
        return "audit ok"

    assert audit("admin") == "audit ok"


# ---------------------------------------------------------------------------------------------------
# TEST: ADD USER / UPDATE USER
# ---------------------------------------------------------------------------------------------------
def test_add_or_update_user_by_admin():
    auth_manager.add_or_update_user("admin", "new_user", "viewer")
    users = load_users()
    assert users["new_user"] == "viewer"


def test_add_or_update_user_denied():
    with pytest.raises(PermissionError):
        auth_manager.add_or_update_user("viewer_user", "hacker", "admin")


def test_update_existing_user():
    auth_manager.add_or_update_user("admin", "alice_reviewer", "auditor")
    users = load_users()
    assert users["alice_reviewer"] == "auditor"


# ---------------------------------------------------------------------------------------------------
# TEST: REMOVE USER
# ---------------------------------------------------------------------------------------------------
def test_remove_user_admin_only():
    auth_manager.remove_user("admin", "viewer_user")
    users = load_users()
    assert "viewer_user" not in users


def test_remove_user_denied():
    with pytest.raises(PermissionError):
        auth_manager.remove_user("viewer_user", "alice_reviewer")


def test_remove_user_missing_ok():
    # Should not crash when deleting non-existing user
    assert auth_manager.remove_user("admin", "ghost_user") is True


# ---------------------------------------------------------------------------------------------------
# TEST: FILE AUTO-CREATION ON MISSING FILE
# ---------------------------------------------------------------------------------------------------
def test_load_users_when_file_missing(monkeypatch, tmp_path):
    """
    If users.json does not exist, auth_manager should fall back to DEFAULT_USERS.
    """
    missing_path = tmp_path / "missing_users.json"
    monkeypatch.setattr(auth_manager, "USERS_PATH", str(missing_path))

    # call loader → should use default users
    users = auth_manager._load_users()
    assert users["admin"] == "admin"
    assert users["alice_reviewer"] == "reviewer"
