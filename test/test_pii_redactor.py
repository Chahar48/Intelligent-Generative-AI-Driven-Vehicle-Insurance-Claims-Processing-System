# test/test_pii_redactor.py

import pytest
from claim_pipeline.security import pii_redactor as R


# ----------------------------------------------------------------------------------------------------
# EMAIL REDACTION
# ----------------------------------------------------------------------------------------------------
def test_redact_email_basic():
    assert R.redact_email("john.doe@gmail.com").startswith("jo")
    masked = R.redact_email("john.doe@gmail.com")
    assert masked.endswith("@gmail.com")
    assert "*" in masked


def test_redact_email_short_user():
    assert R.redact_email("ab@gmail.com") == "a*@gmail.com"


def test_redact_email_no_match():
    assert R.redact_email("not-email") == "not-email"


def test_redact_email_none():
    assert R.redact_email(None) is None


# ----------------------------------------------------------------------------------------------------
# PHONE REDACTION
# ----------------------------------------------------------------------------------------------------
def test_redact_phone_basic():
    masked = R.redact_phone("+91-9876-543210")
    assert masked.startswith("98")
    assert masked.endswith("10")
    assert "*" in masked


def test_redact_phone_small_number():
    assert R.redact_phone("1234") == "****"  # all masked


def test_redact_phone_none():
    assert R.redact_phone(None) is None


# ----------------------------------------------------------------------------------------------------
# CLAIM / POLICY ID
# ----------------------------------------------------------------------------------------------------
def test_redact_claim_id():
    text = "CLM-998877"
    masked = R.redact_claim_or_policy_id(text)
    assert masked != text
    assert masked.startswith("CLM")
    assert "*" in masked


def test_redact_policy_id_in_sentence():
    t = "Policy No: PN-123456 is active"
    out = R.redact_claim_or_policy_id(t)
    assert "PN-" in out
    assert "*" in out


def test_redact_claim_no_match():
    assert R.redact_claim_or_policy_id("nothing to redact") == "nothing to redact"


# ----------------------------------------------------------------------------------------------------
# NAME REDACTION
# ----------------------------------------------------------------------------------------------------
def test_redact_name_two_words():
    assert R.redact_name("John Doe") == "John D***"


def test_redact_name_single_word():
    assert R.redact_name("John") == "J***"


def test_redact_name_none():
    assert R.redact_name(None) is None


# ----------------------------------------------------------------------------------------------------
# TEXT REDACTION (ORDERED PIPELINE)
# ----------------------------------------------------------------------------------------------------
def test_redact_text_all_types():
    text = """
        John Doe contacted at john.doe@gmail.com
        Phone: +91-98765-43210
        Claim: CLM-998877
    """

    out = R.redact_text(text)

    assert "John" in out and "*" in out  # name masked
    assert "@gmail.com" in out  # domain visible
    assert "CLM-" in out and "*" in out  # claim masked
    assert "987" in out and "*" in out  # phone masked


def test_redact_text_none():
    assert R.redact_text(None) is None


def test_redact_text_no_pii():
    assert R.redact_text("hello world") == "hello world"


# ----------------------------------------------------------------------------------------------------
# REDACT_DICT - SIMPLE CASES
# ----------------------------------------------------------------------------------------------------
def test_redact_dict_basic_fields():
    data = {
        "email": "john.doe@gmail.com",
        "phone": "+91-9876543210",
        "claim_id": "CLM-123456",
        "name": "John Doe",
    }

    out = R.redact_dict(data)

    assert out["email"] != "john.doe@gmail.com"
    assert "*" in out["phone"]
    assert "*" in out["claim_id"]
    assert "John" in out["name"]  # masked last name


def test_redact_dict_nested_structures():
    data = {
        "customer": {
            "email": "john@gmail.com",
            "contact": "9876543210",
            "details": {
                "policy_no": "PN-778899"
            }
        }
    }

    out = R.redact_dict(data)

    assert "*" in out["customer"]["email"]
    assert "*" in out["customer"]["contact"]
    assert "*" in out["customer"]["details"]["policy_no"]


def test_redact_dict_list():
    data = [
        {"email": "john@gmail.com"},
        {"phone": "9999988888"}
    ]

    out = R.redact_dict(data)

    assert "*" in out[0]["email"]
    assert "*" in out[1]["phone"]


def test_redact_dict_string_input():
    text = "John Doe email john@gmail.com"
    out = R.redact_dict(text)
    assert "*" in out  # email masked


# ----------------------------------------------------------------------------------------------------
# REDACT_DICT - KEYS TO KEEP (NO REDACTION FOR SPECIFIED KEYS)
# ----------------------------------------------------------------------------------------------------
def test_redact_dict_keys_to_keep():
    data = {
        "email": "john.doe@gmail.com",
        "phone": "9999988888",
        "claim_id": "CLM-123456",
        "notes": "John Doe has email john@aaa.com",
    }

    out = R.redact_dict(data, keys_to_keep={"email": True, "claim_id": True})

    # kept as-is
    assert out["email"] == "john.doe@gmail.com"
    assert out["claim_id"] == "CLM-123456"

    # redacted normally
    assert "*" in out["phone"]
    assert "*" in out["notes"]  # name/email inside notes should still be masked


# ----------------------------------------------------------------------------------------------------
# EDGE CASES
# ----------------------------------------------------------------------------------------------------
def test_redact_dict_empty():
    assert R.redact_dict({}) == {}
    assert R.redact_dict([]) == []


def test_redact_dict_non_string_non_collection():
    assert R.redact_dict(12345) == 12345
    assert R.redact_dict(12.34) == 12.34
    assert R.redact_dict(True) is True
