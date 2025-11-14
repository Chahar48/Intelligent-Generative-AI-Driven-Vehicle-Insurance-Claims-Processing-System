# test/test_field_extractor.py

import pytest
from claim_pipeline.processing.field_extractor import extract_claim_fields


# ----------------------------------------------------------------------
# Test: EMPTY TEXT
# ----------------------------------------------------------------------
def test_empty_text():
    res = extract_claim_fields("")
    assert res["fields"] == {}
    assert "EMPTY_TEXT" in res["notes"]


# ----------------------------------------------------------------------
# Test: Basic extraction of high-confidence regex fields
# ----------------------------------------------------------------------
def test_basic_field_extraction():
    text = """
    Claim ID: CLM-77889
    Policy Number: P-112233
    Name: John Doe
    Email: john.doe@example.com
    Phone: +91 98765 43210
    Date: 2024-10-12
    """

    res = extract_claim_fields(text)
    f = res["fields"]

    assert f["claim_id"]["value"] == "CLM-77889"
    assert f["policy_no"]["value"] == "P-112233"
    assert f["claimant_name"]["value"] == "John Doe"
    assert f["email"]["value"] == "john.doe@example.com"
    assert f["phone"]["value"] == "+91 98765 43210"
    assert f["date"]["value"] == "2024-10-12"

    assert f["claim_id"]["method"] == "regex"
    assert f["email"]["method"] == "regex"


# ----------------------------------------------------------------------
# Test: Multiline vehicle extraction
# ----------------------------------------------------------------------
def test_vehicle_multiline_extraction():
    text = """
    Vehicle: Maruti Suzuki Swift
      Colour: Red
      Registration: RJ14 XX 0099
    """

    res = extract_claim_fields(text)
    f = res["fields"]

    assert "Maruti Suzuki Swift" in f["vehicle"]["value"]
    assert "Registration" in f["vehicle"]["value"]
    assert f["vehicle"]["method"] == "regex-multiline"


# ----------------------------------------------------------------------
# Test: Multiline damage description
# ----------------------------------------------------------------------
def test_damage_multiline_extraction():
    text = """
    Damage Description: Front bumper cracked
       left indicator broken
       scratches on right door
    """

    res = extract_claim_fields(text)
    f = res["fields"]

    assert "Front bumper cracked" in f["damage"]["value"]
    assert "indicator" in f["damage"]["value"]
    assert "scratches" in f["damage"]["value"]
    assert f["damage"]["method"] == "regex-multiline"


# ----------------------------------------------------------------------
# Test: Amount extraction - strict regex
# ----------------------------------------------------------------------
def test_amount_strict():
    text = """
    The total estimated repair is Rs. 15,500.
    """

    res = extract_claim_fields(text)
    f = res["fields"]

    assert f["amount_estimated"]["value"] in ["Rs. 15,500", "Rs. 15,500."]


# ----------------------------------------------------------------------
# Test: Estimate extraction - estimate regex
# ----------------------------------------------------------------------
def test_amount_estimate_regex():
    text = """
    Estimated Amount: ₹ 22,340
    """

    res = extract_claim_fields(text)
    f = res["fields"]

    assert "22,340" in f["amount_estimated"]["value"]
    assert f["amount_estimated"]["method"] == "estimate-regex"


# ----------------------------------------------------------------------
# Test: Amount heuristic fallback
# ----------------------------------------------------------------------
def test_amount_heuristic():
    text = """
    Repair charges approx: $1200
    """

    res = extract_claim_fields(text)
    f = res["fields"]

    assert "$1200" in f["amount_estimated"]["value"]
    assert f["amount_estimated"]["method"] == "amount-heuristic"


# ----------------------------------------------------------------------
# Test: Missing fields produce notes
# ----------------------------------------------------------------------
def test_missing_fields_notes():
    text = "Random text without fields"

    res = extract_claim_fields(text)

    missing_keys = {
        "claim_id",
        "policy_no",
        "claimant_name",
        "phone",
        "email",
        "vehicle",
        "damage",
        "amount_estimated",
        "date",
    }

    for key in missing_keys:
        assert f"missing:{key}" in res["notes"]
