import pytest
import json

from claim_pipeline.processing.decision_engine import (
    decide_claim,
    rule_based_decision,
    _extract_json_from_text,
    _safe_json_load,
)

# -----------------------------
# FIXTURES
# -----------------------------
@pytest.fixture
def base_fields_struct():
    """Minimal realistic structured fields used across tests."""
    return {
        "claim_id": {"value": "CLM-1001", "canonical": "CLM-1001", "score": 0.9},
        "policy_no": {"value": "PN-5555", "canonical": "PN-5555", "score": 0.9},
        "claimant_name": {"value": "John Doe", "canonical": "John Doe", "score": 0.9},
        "phone": {"value": "+919876543210", "canonical": "+919876543210", "score": 0.9},
        "email": {"value": "john@example.com", "canonical": "john@example.com", "score": 0.9},
        "vehicle": {"value": "Honda City", "canonical": "Honda City", "score": 0.8},
        "damage": {"value": "Front bumper", "canonical": "Front bumper", "score": 0.8},
        "date": {"value": "2024-10-12", "canonical": "2024-10-12", "score": 0.95},
        "amount_estimated": {"value": 15000.0, "canonical": "INR 15000.00", "score": 0.9},
        "insured_amount": {"value": 20000.0, "canonical": "INR 20000.00", "score": 0.9},
    }

@pytest.fixture
def base_validation():
    """Validates minimal structured fields."""
    return {
        "issues": [],
        "warnings": [],
        "missing_required": [],
        "recommendation": "approve",
        "fields": {},
        "normalized": {}
    }

# ------------------------------------
# BASIC RULE DECISIONS
# ------------------------------------
def test_rule_based_decision_within_limit(base_fields_struct, base_validation):
    """Claim amount < insured → approve."""
    res = rule_based_decision(base_fields_struct, base_validation)
    assert res["decision"] == "approve"
    assert res["confidence"] >= 0.8
    assert "within_limit" in res["flags"]

def test_rule_based_over_limit(base_fields_struct, base_validation):
    """Claim amount > insured → reject."""
    fields = base_fields_struct.copy()
    fields["amount_estimated"] = {"value": 50000.0, "canonical": "INR 50000.00", "score": 0.9}
    fields["insured_amount"] = {"value": 20000.0, "canonical": "INR 20000.00", "score": 0.9}

    res = rule_based_decision(fields, base_validation)
    assert res["decision"] == "reject"
    assert "over_limit" in res["flags"]
    assert res["confidence"] >= 0.9

def test_rule_based_missing_required_fields(base_fields_struct, base_validation):
    """Missing required fields should trigger review."""
    validation = base_validation.copy()
    validation["missing_required"] = ["policy_no"]
    res = rule_based_decision(base_fields_struct, validation)

    assert res["decision"] == "review"
    assert "missing_fields" in res["flags"]

def test_rule_based_high_risk_keyword(base_fields_struct, base_validation):
    """Damage containing high-risk triggers review."""
    fields = base_fields_struct.copy()
    fields["damage"] = {
        "value": "Total loss of the vehicle",
        "canonical": "Total loss of the vehicle",
        "score": 0.9
    }

    res = rule_based_decision(fields, base_validation)
    assert res["decision"] == "review"
    assert "high_risk_keyword" in res["flags"]

def test_rule_based_low_confidence_penalty(base_fields_struct, base_validation):
    """Low confidence field reduces decision confidence."""
    fields = base_fields_struct.copy()
    fields["email"]["score"] = 0.3

    res = rule_based_decision(fields, base_validation)
    assert "low_confidence_fields" in res["flags"]
    assert res["confidence"] <= 0.75

# ------------------------------------
# DECISION ENGINE TOP-LEVEL
# ------------------------------------
def test_decide_claim_rule_only(base_fields_struct, base_validation, monkeypatch):
    """If Groq unavailable → rule-only path."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    res = decide_claim(base_fields_struct, base_validation)

    assert res["source"] == "rule"
    assert "decision" in res
    assert "reason" in res
    assert "confidence" in res

def test_decide_claim_force_ai_but_no_api(base_fields_struct, base_validation, monkeypatch):
    """force_ai=True but no Groq → rule result."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    res = decide_claim(base_fields_struct, base_validation, force_ai=True)

    assert res["source"] == "rule"
    assert "decision" in res

# ------------------------------------
# AI REASONING (MOCKED)
# ------------------------------------
def test_decide_claim_ai_override(base_fields_struct, base_validation, monkeypatch):
    """Mock Groq to return JSON → decision overrides rule."""

    # Mock Groq client & response
    class MockChoice:
        def __init__(self, content): self.message = type("msg", (), {"content": content})

    class MockResp:
        def __init__(self, content): self.choices = [MockChoice(content)]

    class MockGroq:
        def chat(self): return None
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature, max_tokens):
                    return MockResp('{ "decision": "reject", "reason": "AI sees fraud", "confidence": 0.95 }')

    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    monkeypatch.setattr("claim_pipeline.processing.decision_engine.llm_client", MockGroq())

    # Set conditions that normally lead to "approve"
    res = decide_claim(base_fields_struct, base_validation, summary={"dummy": True}, force_ai=True)

    assert res["decision"] == "reject"
    assert res["source"] == "rule+ai"
    assert res["confidence"] >= 0.95

# ------------------------------------
# JSON EXTRACTION UNIT TESTS
# ------------------------------------
def test_extract_json_basic():
    txt = "Random text\n```json\n{ \"a\": 1 }\n``` more"
    out = _extract_json_from_text(txt)
    assert out == '{ "a": 1 }'

def test_extract_json_nested():
    txt = """
    some text {
        "a": 1,
        "b": {"c":2}
    } trailing
    """
    out = _extract_json_from_text(txt)
    data = json.loads(out)
    assert data["b"]["c"] == 2

def test_safe_json_load_repair():
    """Tests repairing malformed JSON."""
    s = "{ 'a': 1, }"
    out = _safe_json_load(s)
    assert out["a"] == 1

# ------------------------------------
# EDGE CASES
# ------------------------------------
def test_decision_missing_amounts(base_fields_struct, base_validation):
    fields = base_fields_struct.copy()
    fields["amount_estimated"]["value"] = None

    res = rule_based_decision(fields, base_validation)
    assert "no_claim_amount" in res["flags"]
    assert res["decision"] in ("review", "approve")  # Depending on insured amt

def test_decision_missing_everything():
    fields = {}
    validation = {
        "issues": [],
        "warnings": [],
        "missing_required": ["claim_id", "date"],
        "recommendation": "review",
        "fields": {},
        "normalized": {}
    }

    res = rule_based_decision(fields, validation)
    assert res["decision"] == "review"
    assert "missing_fields" in res["flags"]
    assert res["confidence"] <= 0.6
