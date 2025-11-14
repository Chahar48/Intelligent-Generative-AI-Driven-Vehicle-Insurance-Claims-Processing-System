# test/test_summarizer.py

import pytest
from claim_pipeline.ai.summarizer import (
    generate_claim_summary,
    _extract_json_from_text,
    _safe_json_load,
)

# -----------------------------------------------------------
# Fixtures
# -----------------------------------------------------------

@pytest.fixture
def raw_text():
    return """
    Claim ID: CLM-8899
    Policy No: PN-5522
    Vehicle: Honda City ZX
    Estimated Amount: Rs. 25,000
    Damage Description: Front bumper cracked
    Date: 2024-10-10
    """


@pytest.fixture
def fields_struct():
    return {
        "claim_id": {"value": "CLM-8899", "canonical": "CLM-8899", "confidence": "high"},
        "policy_no": {"value": "PN-5522", "canonical": "PN-5522", "confidence": "high"},
        "vehicle": {"value": "Honda City ZX", "canonical": "Honda City ZX", "confidence": "high"},
        "amount_estimated": {"value": 25000.0, "canonical": "INR 25000.00", "confidence": "high"},
        "damage": {"value": "Front bumper cracked", "canonical": "Front bumper cracked", "confidence": "high"},
        "date": {"value": "2024-10-10", "canonical": "2024-10-10", "confidence": "high"},
    }


@pytest.fixture
def validation_ok():
    return {
        "issues": [],
        "missing_required": [],
        "is_complete": True,
        "recommendation": "approve"
    }


@pytest.fixture
def validation_with_errors():
    return {
        "issues": ["amount_missing"],
        "missing_required": ["policy_no"],
        "is_complete": False,
        "recommendation": "review"
    }


# -----------------------------------------------------------
# JSON extraction tests
# -----------------------------------------------------------

def test_extract_json_from_text_basic():
    text = """
    Here is your JSON:
    {
        "a": 1,
        "b": 2
    }
    Thanks.
    """
    extracted = _extract_json_from_text(text)
    assert extracted.strip().startswith("{")
    assert extracted.strip().endswith("}")


def test_extract_json_from_markdown_fence():
    text = """```json
    {"hello": "world"}
    ```"""
    extracted = _extract_json_from_text(text)
    assert extracted == '{"hello": "world"}'


def test_safe_json_load_valid():
    js = '{"x": 10, "y": 20}'
    parsed = _safe_json_load(js)
    assert parsed["x"] == 10
    assert parsed["y"] == 20


def test_safe_json_load_with_single_quotes():
    js = "{'x': 10, 'y': 20}"
    parsed = _safe_json_load(js)
    assert parsed["x"] == 10


# -----------------------------------------------------------
# Summarizer fallback tests
# -----------------------------------------------------------

def test_summarizer_fallback_summary_no_api(raw_text, fields_struct, validation_ok, monkeypatch):
    """If GROQ_API_KEY is missing, summarizer MUST use fallback."""
    monkeypatch.setenv("GROQ_API_KEY", "")

    res = generate_claim_summary(raw_text, fields_struct, validation_ok)

    assert res["source"] == "rule"
    assert "summary" in res
    assert "decision" in res
    assert res["decision"] == "approve"


def test_summarizer_fallback_on_invalid_llm_output(raw_text, fields_struct, validation_ok, monkeypatch):
    """Even if Groq is 'available', invalid JSON must trigger fallback."""
    # Pretend API key exists
    monkeypatch.setenv("GROQ_API_KEY", "dummy123")

    # Mock Groq client to output garbage
    from claim_pipeline.ai import summarizer
    summarizer.llm_available = True
    summarizer.llm_client = None  # Force fallback

    res = generate_claim_summary(raw_text, fields_struct, validation_ok)
    assert res["source"] == "rule"


# -----------------------------------------------------------
# LLM JSON valid output tests
# -----------------------------------------------------------

def test_summarizer_valid_llm_json(raw_text, fields_struct, validation_ok, monkeypatch):
    """Mock Groq to return valid JSON so summarizer returns LLM output."""
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key")

    from claim_pipeline.ai import summarizer
    summarizer.llm_available = True

    class FakeGroqResp:
        class FakeChoice:
            class FakeMessage:
                content = """
                {
                    "summary": "Vehicle Honda City claim.",
                    "decision": "approve",
                    "reasoning": "All details valid.",
                    "confidence": 0.92
                }
                """
            message = FakeMessage()

        choices = [FakeChoice()]

    class FakeGroqClient:
        def chat(self):
            return self

        class completions:
            @staticmethod
            def create(model, messages, temperature, max_tokens):
                return FakeGroqResp()

    summarizer.llm_client = FakeGroqClient()

    res = generate_claim_summary(raw_text, fields_struct, validation_ok)

    assert res["source"] == "llm"
    assert res["decision"] == "approve"
    assert res["confidence"] == 0.92


# -----------------------------------------------------------
# Structure validation
# -----------------------------------------------------------

def test_summary_structure(raw_text, fields_struct, validation_ok, monkeypatch):
    """Ensure returned dict always contains expected keys."""
    monkeypatch.setenv("GROQ_API_KEY", "")

    res = generate_claim_summary(raw_text, fields_struct, validation_ok)

    assert "summary" in res
    assert "decision" in res
    assert "reasoning" in res
    assert "confidence" in res
    assert "source" in res


# -----------------------------------------------------------
# Edge cases
# -----------------------------------------------------------

def test_summary_missing_raw_text(fields_struct, validation_ok):
    """If raw_text is empty → fallback."""
    res = generate_claim_summary("", fields_struct, validation_ok)
    assert res["source"] == "rule"


def test_summary_missing_fields_struct(raw_text, validation_ok):
    """Missing fields struct → fallback."""
    res = generate_claim_summary(raw_text, None, validation_ok)
    assert res["source"] == "rule"


def test_summary_missing_required_fields_impact(raw_text, fields_struct, validation_with_errors, monkeypatch):
    """Ensure fallback includes missing fields in reasoning."""
    monkeypatch.setenv("GROQ_API_KEY", "")

    res = generate_claim_summary(raw_text, fields_struct, validation_with_errors)

    assert "Missing required fields" in res["reasoning"]
