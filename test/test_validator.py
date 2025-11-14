import pytest
from datetime import datetime, timedelta
from claim_pipeline.processing.validator import validate_claim_fields


# -----------------------------------------------------------
# Helper function: A fully valid claim
# -----------------------------------------------------------
def valid_claim():
    return {
        "claim_id": "CLM001",
        "policy_no": "POL123",
        "claimant_name": "John Doe",
        "phone": "+91 9876543210",
        "email": "john.doe@example.com",
        "vehicle": "Honda City",
        "damage": "Front bumper crack",
        "date": "2024-10-12",
        "amount_estimated": "Rs. 15,500",
        "insured_amount": "Rs. 20000",
    }


# -----------------------------------------------------------
# 1. FULLY VALID CLAIM → APPROVE
# -----------------------------------------------------------
def test_validator_valid_claim():
    data = valid_claim()
    result = validate_claim_fields(data)

    assert result["is_complete"] is True
    assert result["recommendation"] == "approve"
    assert result["risk_score"] < 0.5
    assert "amount_estimated_exceeds_insured_amount" not in result["issues"]
    assert result["fields"]["amount_estimated"]["value"] == 15500.0


# -----------------------------------------------------------
# 2. Amount exceeds insured → REJECT
# -----------------------------------------------------------
def test_validator_amount_exceeds_insured_amount():
    data = valid_claim()
    data["amount_estimated"] = "Rs. 35000"  # More than insured amount

    result = validate_claim_fields(data)

    assert "amount_estimated_exceeds_insured_amount" in result["issues"]
    assert result["recommendation"] == "reject"


# -----------------------------------------------------------
# 3. Missing required field → REVIEW
# -----------------------------------------------------------
def test_validator_missing_required_field():
    data = valid_claim()
    data["policy_no"] = None

    result = validate_claim_fields(data)

    assert "policy_no" in result["missing_required"]
    assert result["recommendation"] == "review"
    assert result["is_complete"] is False


# -----------------------------------------------------------
# 4. Invalid date → AI suggestion + parse_failed
# -----------------------------------------------------------
def test_validator_invalid_date():
    data = valid_claim()
    data["date"] = "abc xyz"

    result = validate_claim_fields(data)

    assert "date" in result["ai_fields"]
    assert "date:parse_failed" in result["notes"]


# -----------------------------------------------------------
# 5. Future date → warning
# -----------------------------------------------------------
def test_validator_future_date():
    data = valid_claim()
    future = (datetime.utcnow() + timedelta(days=10)).date().isoformat()
    data["date"] = future

    result = validate_claim_fields(data)

    assert "date_in_future" in result["warnings"]


# -----------------------------------------------------------
# 6. Very old date (>5 years) → warning
# -----------------------------------------------------------
def test_validator_old_date():
    data = valid_claim()
    old_date = (datetime.utcnow() - timedelta(days=365 * 6)).date().isoformat()
    data["date"] = old_date

    result = validate_claim_fields(data)

    assert "date_old" in result["warnings"]


# -----------------------------------------------------------
# 7. Suspicious phone → warning
# -----------------------------------------------------------
def test_validator_suspicious_phone():
    data = valid_claim()
    data["phone"] = "0000000000"

    result = validate_claim_fields(data)

    assert "phone_suspicious" in result["warnings"]


# -----------------------------------------------------------
# 8. Suspicious email domain
# -----------------------------------------------------------
def test_validator_suspicious_email_domain():
    data = valid_claim()
    data["email"] = "john@fake.cm"

    result = validate_claim_fields(data)

    assert "email_domain_suspicious" in result["warnings"]


# -----------------------------------------------------------
# 9. Invalid email → AI field suggestion
# -----------------------------------------------------------
def test_validator_invalid_email():
    data = valid_claim()
    data["email"] = "invalid-email"

    result = validate_claim_fields(data)

    assert "email" in result["ai_fields"]
    assert result["fields"]["email"]["confidence"] == "low"


# -----------------------------------------------------------
# 10. Check validator response structure
# -----------------------------------------------------------
def test_validator_structure():
    data = valid_claim()
    result = validate_claim_fields(data)

    expected_keys = [
        "fields", "issues", "warnings", "notes", "missing_required",
        "ai_fields", "risk_score", "is_complete", "recommendation",
        "decision_reasons", "explainability",
    ]

    for key in expected_keys:
        assert key in result


# -----------------------------------------------------------
# 11. Explainability contains field_scores
# -----------------------------------------------------------
def test_validator_explainability_field_scores():
    data = valid_claim()
    result = validate_claim_fields(data)

    assert "field_scores" in result["explainability"]
    assert isinstance(result["explainability"]["field_scores"], dict)


# -----------------------------------------------------------
# 12. Phone autocorrect ("0" prefix)
# -----------------------------------------------------------
def test_validator_phone_autocorrect():
    data = valid_claim()
    data["phone"] = "09876543210"

    result = validate_claim_fields(data)

    assert "phone:auto_corrected" in result["warnings"]
    assert result["fields"]["phone"]["corrected"] is True


# -----------------------------------------------------------
# 13. Invalid amount → no_numeric_found
# -----------------------------------------------------------
def test_validator_invalid_amount():
    data = valid_claim()
    data["amount_estimated"] = "no value"

    result = validate_claim_fields(data)

    assert "amount_estimated:no_numeric_found" in result["notes"]
    assert result["fields"]["amount_estimated"]["value"] is None


# -----------------------------------------------------------
# 14. Risk score grows when errors increase
# -----------------------------------------------------------
def test_validator_risk_score_increases():
    data = valid_claim()
    data["policy_no"] = None
    data["amount_estimated"] = "no number"

    result = validate_claim_fields(data)

    assert result["risk_score"] >= 0.3
    assert result["recommendation"] in ("review", "reject")


# -----------------------------------------------------------
# 15. Multiple candidates (simulated)
# -----------------------------------------------------------
def test_validator_multiple_candidates_simulated():
    data = valid_claim()
    result = validate_claim_fields(data)

    # simulate normalizer producing multiple values
    result["fields"]["policy_no"]["raw_candidates"] = ["POL1", "POL2"]

    assert len(result["fields"]["policy_no"]["raw_candidates"]) == 2
