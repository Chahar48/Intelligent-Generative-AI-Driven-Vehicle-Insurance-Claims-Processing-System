import pytest
import json
from claim_pipeline.processing.ai_field_infer import (
    infer_claim_fields,
    _prepare_context,
    _extract_json_from_text,
    _safe_json_load,
    _hash_prompt,
    EXPECTED_FIELDS,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def raw_text():
    return """
    Claim ID: CLM-77889
    Policy Number: PN-991122
    Name: John Doe
    Phone: +91 9876543210
    Email: john.doe@example.com
    Vehicle: Honda City VMT
    Damage Description: Front bumper cracked
    Amount Estimated: Rs. 15,500
    Date: 2024-10-12
    """


@pytest.fixture
def partial_fields():
    """Simulate validator-normalizer output with missing and low-confidence fields."""
    return {
        "claim_id": {"value": "CLM-77889", "score": 0.9, "canonical": "CLM-77889"},
        "policy_no": {"value": None, "score": 0.1, "canonical": None},
        "claimant_name": {"value": None, "score": 0.0, "canonical": None},
        "phone": {"value": "+919876543210", "score": 0.8, "canonical": "+919876543210"},
        "email": {"value": "john.doe@example.com", "score": 0.9, "canonical": "john.doe@example.com"},
        "vehicle": {"value": None, "score": 0.2, "canonical": None},
        "damage": {"value": None, "score": 0.1, "canonical": None},
        "amount_estimated": {"value": 15500.0, "score": 0.9, "canonical": "INR 15500.00"},
        "date": {"value": "2024-10-12", "score": 0.8, "canonical": "2024-10-12"},
        "insured_amount": {"value": None, "score": 0.0},
        "raw_lines": [
            {"index": 0, "text": "Claim ID: CLM-77889"},
            {"index": 1, "text": "Policy Number: PN-991122"},
        ],
    }


# --------------------------------------------------------------------------
# 1. Context Builder
# --------------------------------------------------------------------------
def test_prepare_context_uses_raw_lines(partial_fields, raw_text):
    ctx = _prepare_context(raw_text, partial_fields, max_lines=5)
    assert "Claim ID:" in ctx
    assert "Policy Number:" in ctx


def test_prepare_context_falls_back_to_raw_text(raw_text):
    ctx = _prepare_context(raw_text, {}, max_lines=3)
    assert "Claim ID" in ctx


# --------------------------------------------------------------------------
# 2. JSON Extraction & Safe Loader
# --------------------------------------------------------------------------
def test_extract_json_from_text_clean_block():
    txt = "Here is output:\n{\"a\":1, \"b\":2}"
    js = _extract_json_from_text(txt)
    assert js == "{\"a\":1, \"b\":2}"


def test_extract_json_from_text_with_markdown():
    txt = "```json\n{\"a\":2, \"b\":3}\n```"
    js = _extract_json_from_text(txt)
    assert json.loads(js)["a"] == 2


def test_safe_json_load_repairs_single_quotes():
    s = "{'a': 1, 'b':2}"
    js = _safe_json_load(s)
    assert js["a"] == 1


# --------------------------------------------------------------------------
# 3. Prompt hash stability
# --------------------------------------------------------------------------
def test_hash_prompt_stable():
    p = "test prompt"
    h1 = _hash_prompt(p)
    h2 = _hash_prompt(p)
    assert h1 == h2
    assert len(h1) == 64   # SHA-256 hex


# --------------------------------------------------------------------------
# 4. Rule-based inference end-to-end (Groq disabled)
# --------------------------------------------------------------------------
def test_infer_claim_fields_rule_mode(raw_text, partial_fields, monkeypatch):
    """Force rule mode."""
    monkeypatch.setenv("AI_OFFLINE_STUB", "true")

    res = infer_claim_fields(raw_text, partial_fields, use_groq=False)

    # Should return shaped dict for all EXPECTED_FIELDS
    for f in EXPECTED_FIELDS:
        assert f in res
        assert "value" in res[f]
        assert "confidence" in res[f]
        assert "source" in res[f]

    # Check fallback inference happened for missing fields
    assert res["policy_no"]["source"] == "rule"
    assert res["claimant_name"]["source"] in ("rule", "partial", "llm")


# --------------------------------------------------------------------------
# 5. Partial fields preserved when valid
# --------------------------------------------------------------------------
def test_infer_keeps_partial_if_high_score(raw_text, partial_fields, monkeypatch):
    monkeypatch.setenv("AI_OFFLINE_STUB", "true")

    res = infer_claim_fields(raw_text, partial_fields, fields_to_infer=["claim_id"], use_groq=False)

    assert res["claim_id"]["value"] == "CLM-77889"
    assert res["claim_id"]["source"] == "partial"


# --------------------------------------------------------------------------
# 6. Missing → rule-based inferred values
# --------------------------------------------------------------------------
def test_infer_missing_field(raw_text, partial_fields, monkeypatch):
    monkeypatch.setenv("AI_OFFLINE_STUB", "true")

    partial_fields["vehicle"]["value"] = None

    res = infer_claim_fields(raw_text, partial_fields, fields_to_infer=["vehicle"], use_groq=False)

    assert res["vehicle"]["source"] == "rule"
    assert res["vehicle"]["value"] is not None  # rule should infer from text


# --------------------------------------------------------------------------
# 7. AI change log exists & records differences
# --------------------------------------------------------------------------
def test_ai_change_log_present(raw_text, partial_fields, monkeypatch):
    monkeypatch.setenv("AI_OFFLINE_STUB", "true")

    res = infer_claim_fields(raw_text, partial_fields, use_groq=False)

    assert "_ai_change_log" in res
    assert isinstance(res["_ai_change_log"], list)


# --------------------------------------------------------------------------
# 8. Shaping & structure validation
# --------------------------------------------------------------------------
def test_infer_output_structure(raw_text, partial_fields, monkeypatch):
    monkeypatch.setenv("AI_OFFLINE_STUB", "true")
    res = infer_claim_fields(raw_text, partial_fields, use_groq=False)

    for f in EXPECTED_FIELDS:
        field = res[f]
        assert isinstance(field, dict)
        assert set([
            "original", "value", "canonical", "confidence",
            "score", "corrected", "notes", "metadata",
            "source", "explain"
        ]).issubset(set(field.keys()))


# --------------------------------------------------------------------------
# 9. Cache behavior
# --------------------------------------------------------------------------
def test_cache_hit(raw_text, partial_fields, monkeypatch):
    monkeypatch.setenv("AI_OFFLINE_STUB", "true")
    # First call
    r1 = infer_claim_fields(raw_text, partial_fields, use_groq=False)
    # Second call — must hit cache
    r2 = infer_claim_fields(raw_text, partial_fields, use_groq=False)

    assert r1 == r2   # Should be identical due to cache


# --------------------------------------------------------------------------
# 10. Ensure no crash when everything missing
# --------------------------------------------------------------------------
def test_infer_all_fields_missing(monkeypatch):
    monkeypatch.setenv("AI_OFFLINE_STUB", "true")

    raw_text = "Some random text"
    partial = {f: {"value": None, "score": 0.0} for f in EXPECTED_FIELDS}

    res = infer_claim_fields(raw_text, partial, use_groq=False)

    assert set(res.keys()) >= set(EXPECTED_FIELDS)
