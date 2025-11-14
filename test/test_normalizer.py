import pytest
from claim_pipeline.processing import normalizer
from datetime import datetime


# -----------------------------------------------------------------------------------
# normalize_amount
# -----------------------------------------------------------------------------------
def test_normalize_amount_inr():
    res = normalizer.normalize_amount("Rs. 15,500")
    assert res["value"] == 15500.0
    assert res["canonical"] in ["INR 15500.00", "INR 15500.00"]
    assert res["confidence"] in ["high", "medium"]


def test_normalize_amount_usd():
    res = normalizer.normalize_amount("$1200.50")
    assert res["value"] == 1200.50
    assert "USD" in res["canonical"]


def test_normalize_amount_with_ocr_noise():
    res = normalizer.normalize_amount("R5. l5,5OO")
    assert res["value"] == 15500.0


def test_normalize_amount_invalid():
    res = normalizer.normalize_amount("no money here")
    assert res["value"] is None
    assert "no_numeric_found" in res["notes"]


def test_normalize_amount_large_invalid():
    res = normalizer.normalize_amount("Rs. 999999999999999")
    assert res["value"] is None
    assert "amount_exceeds_upper_bound" in res["notes"]


# -----------------------------------------------------------------------------------
# normalize_date
# -----------------------------------------------------------------------------------
def test_normalize_date_iso():
    res = normalizer.normalize_date("2024-10-12")
    assert res["value"] == "2024-10-12"
    assert res["confidence"] == "high"


def test_normalize_date_slash_format():
    res = normalizer.normalize_date("12/10/2024")
    assert res["value"] in ["2024-12-10", "2024-10-12"]  # depends on dayfirst detection


def test_normalize_date_textual_month():
    res = normalizer.normalize_date("12 Oct 2024")
    assert res["value"] == "2024-10-12"


def test_normalize_date_with_ocr_noise():
    res = normalizer.normalize_date("2O24-lO-l2")  # O instead of 0, l instead of 1
    assert res["value"] == "2024-10-12"
    assert res["corrected"] is True


def test_normalize_date_invalid():
    res = normalizer.normalize_date("Invalid Date")
    assert res["value"] is None
    assert "parse_failed" in res["notes"]


# -----------------------------------------------------------------------------------
# normalize_phone
# -----------------------------------------------------------------------------------
def test_normalize_phone_india_proper():
    res = normalizer.normalize_phone("+91 98765 43210")
    assert res["canonical"] == "+919876543210"
    assert res["confidence"] == "medium"


def test_normalize_phone_india_local():
    res = normalizer.normalize_phone("9876543210")
    assert res["canonical"] == "+919876543210"
    assert res["confidence"] == "high"


def test_normalize_phone_with_zero():
    res = normalizer.normalize_phone("09876543210")
    assert res["canonical"] == "+919876543210"
    assert res["corrected"] is True


def test_normalize_phone_invalid():
    res = normalizer.normalize_phone("12345")
    assert res["value"] is None or res["confidence"] == "low"
    assert "invalid_length" in res["notes"]


# -----------------------------------------------------------------------------------
# normalize_email
# -----------------------------------------------------------------------------------
def test_normalize_email_basic():
    res = normalizer.normalize_email("John.Doe@Example.com")
    assert res["value"] == "john.doe@example.com"
    assert res["confidence"] in ["high", "medium"]


def test_normalize_email_with_ocr_error_domain():
    res = normalizer.normalize_email("john@gmai.com")
    assert res["value"] == "john@gmail.com"
    assert res["corrected"] is True


def test_normalize_email_missing_at_fixes():
    res = normalizer.normalize_email("john.doe (at) gmail.com")
    assert res["value"] == "john.doe@gmail.com"
    assert res["corrected"] is True


def test_normalize_email_invalid():
    res = normalizer.normalize_email("not-an-email")
    assert res["value"] is None
    assert res["confidence"] == "low"


# -----------------------------------------------------------------------------------
# normalize_string
# -----------------------------------------------------------------------------------
def test_normalize_string_basic():
    res = normalizer.normalize_string("Hello   World")
    assert res["value"] == "Hello World"
    assert res["confidence"] == "high"


def test_normalize_string_unicode_cleanup():
    res = normalizer.normalize_string("ﬁle — test")
    assert "fi" in res["value"]
    assert "-" in res["value"]


def test_normalize_string_truncate():
    s = "A" * 3000
    res = normalizer.normalize_string(s, max_len=100)
    assert len(res["value"]) == 100
    assert "truncated_to_100" in res["notes"]
