# claim_pipeline/security/auth_manager.py
"""
Simple Role-Based Access Control for local prototype.

Usage:
- add users to data/users.json (username -> role)
- use has_role(username, "reviewer") to check
- use require_role(username, "admin") to enforce in code

Roles (suggested):
- admin
- reviewer
- auditor
- viewer

Note: This is not a full auth system. In production, integrate with OAuth/JWT/LDAP.
"""

import json
import os
from functools import wraps
from typing import Optional, Callable

USERS_PATH = "data/users.json"
DEFAULT_USERS = {
    "admin": "admin",         # username: role
    "alice_reviewer": "reviewer",
    "bob_auditor": "auditor",
    "viewer_user": "viewer"
}

os.makedirs(os.path.dirname(USERS_PATH) or ".", exist_ok=True)
if not os.path.exists(USERS_PATH):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_USERS, f, indent=2)

def _load_users() -> dict:
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_USERS.copy()

def get_user_role(username: str) -> Optional[str]:
    users = _load_users()
    return users.get(username)

def has_role(username: str, role: str) -> bool:
    user_role = get_user_role(username)
    if not user_role:
        return False
    return user_role == role or (user_role == "admin" and role in ("reviewer", "auditor", "viewer"))

def require_role(role: str):
    """
    Decorator to enforce role on a function with signature func(username, *args, **kwargs)
    Example:
        @require_role("reviewer")
        def submit_review(username, claim_id, ...):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(username: str, *args, **kwargs):
            if not has_role(username, role):
                raise PermissionError(f"User '{username}' lacks required role '{role}'")
            return func(username, *args, **kwargs)
        return wrapper
    return decorator

# Utility to add/remove users (only admin should call in production)
def add_or_update_user(admin_username: str, username: str, role: str):
    if not has_role(admin_username, "admin"):
        raise PermissionError("Only admin can add or update users")
    users = _load_users()
    users[username] = role
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    return True

def remove_user(admin_username: str, username: str):
    if not has_role(admin_username, "admin"):
        raise PermissionError("Only admin can remove users")
    users = _load_users()
    if username in users:
        users.pop(username)
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
    return True
