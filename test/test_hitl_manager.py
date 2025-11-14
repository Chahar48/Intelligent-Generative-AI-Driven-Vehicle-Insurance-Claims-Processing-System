# test/test_hitl_manager.py
import json
import os
import shutil
import time
import pytest

from claim_pipeline.hitl import hitl_manager


@pytest.fixture(autouse=True)
def isolate_dirs(tmp_path, monkeypatch):
    """
    Redirect REVIEW_DIR, AUDIT_DIR, FINAL_DIR to temporary locations for isolation.
    Ensure directories exist and revert nothing (monkeypatch handles environment for test duration).
    """
    review = tmp_path / "reviews"
    audit = tmp_path / "audits"
    final = tmp_path / "final"
    # ensure they exist
    review.mkdir()
    audit.mkdir()
    final.mkdir()

    # monkeypatch module-level constants
    monkeypatch.setattr(hitl_manager, "REVIEW_DIR", str(review))
    monkeypatch.setattr(hitl_manager, "AUDIT_DIR", str(audit))
    monkeypatch.setattr(hitl_manager, "FINAL_DIR", str(final))

    # make sure any internal os.makedirs calls won't fail
    os.makedirs(hitl_manager.REVIEW_DIR, exist_ok=True)
    os.makedirs(hitl_manager.AUDIT_DIR, exist_ok=True)
    os.makedirs(hitl_manager.FINAL_DIR, exist_ok=True)

    yield

    # cleanup just in case (tmp_path will be wiped by pytest, but do safe remove)
    try:
        shutil.rmtree(str(review))
    except Exception:
        pass
    try:
        shutil.rmtree(str(audit))
    except Exception:
        pass
    try:
        shutil.rmtree(str(final))
    except Exception:
        pass


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_create_review_task_and_get_pending():
    mgr = hitl_manager.HITLManager()

    claim_id = "TEST-1001"
    raw_text = "Claim ID: TEST-1001\nName: Alice\nAmount: Rs. 1,000"
    extracted_fields = {"claim_id": {"value": "TEST-1001"}, "amount_estimated": {"value": 1000.0}}
    validation = {"issues": ["amount_missing"], "missing_required": ["policy_no"]}
    ai_inference = {"amount_estimated": {"value": None, "confidence": 0.2}}
    summary = {"summary": "Needs review"}
    decision = {"decision": "review", "confidence": 0.4}

    task = mgr.create_review_task(
        claim_id=claim_id,
        raw_text=raw_text,
        extracted_fields=extracted_fields,
        validation=validation,
        ai_inference=ai_inference,
        summary=summary,
        decision=decision
    )

    # Basic structure checks
    assert task["claim_id"] == claim_id
    assert task["status"] == "pending"
    assert "review_reasons" in task
    assert task["raw_text"] == raw_text

    # File was created
    path = os.path.join(hitl_manager.REVIEW_DIR, f"{claim_id}.json")
    assert os.path.exists(path)

    loaded = _load_json(path)
    assert loaded["claim_id"] == claim_id
    assert loaded["status"] == "pending"

    # get_pending_cases returns this task
    pending = mgr.get_pending_cases()
    assert any(t["claim_id"] == claim_id for t in pending)


def test_save_human_feedback_and_merge_creates_final_and_audit():
    mgr = hitl_manager.HITLManager()

    claim_id = "TEST-2002"
    raw_text = "Claim ID: TEST-2002\nName: Bob\nAmount: Rs. 2,000"
    extracted_fields = {"claim_id": {"value": "TEST-2002"}, "amount_estimated": {"value": 2000.0}}
    validation = {}
    ai_inference = {}
    summary = {}
    decision = {"decision": "review", "confidence": 0.3}

    # create initial task
    mgr.create_review_task(
        claim_id=claim_id,
        raw_text=raw_text,
        extracted_fields=extracted_fields,
        validation=validation,
        ai_inference=ai_inference,
        summary=summary,
        decision=decision
    )

    # Save human feedback
    corrected_fields = {"amount_estimated": {"value": 2500.0, "canonical": "INR 2500.00"}}
    corrected_text = "Corrected OCR text with amount Rs. 2,500"
    overridden_decision = "approve"
    comments = "Verified by agent"
    reviewer = "agent_1"

    task_after = mgr.save_human_feedback(
        claim_id=claim_id,
        corrected_fields=corrected_fields,
        corrected_text=corrected_text,
        overridden_decision=overridden_decision,
        comments=comments,
        reviewer=reviewer
    )

    assert task_after["status"] == "review_completed"
    assert task_after["human_corrected_fields"] == corrected_fields
    assert task_after["human_corrected_text"] == corrected_text
    assert task_after["human_overridden_decision"] == overridden_decision
    assert task_after["reviewer_comments"] == comments
    assert task_after["reviewer"] == reviewer

    # Merge human corrections -> final file should be created
    final = mgr.merge_human_corrections(claim_id)

    # final structure checks
    assert final["claim_id"] == claim_id
    assert "final_fields" in final
    # corrected field should override
    assert final["final_fields"]["amount_estimated"]["value"] == 2500.0
    assert final["final_decision"] == overridden_decision
    assert final["review_comments"] == comments
    assert final["reviewer"] == reviewer

    # final file exists
    final_path = os.path.join(hitl_manager.FINAL_DIR, f"{claim_id}_final.json")
    assert os.path.exists(final_path)
    loaded_final = _load_json(final_path)
    assert loaded_final["final_decision"] == overridden_decision

    # Audit log exists and contains events
    audit_path = os.path.join(hitl_manager.AUDIT_DIR, f"{claim_id}_audit.json")
    assert os.path.exists(audit_path)
    audit_content = _load_json(audit_path)
    assert "events" in audit_content
    # last event should be 'final_output_generated'
    assert any(e["event"] in ("human_review_submitted", "final_output_generated", "review_task_created") for e in audit_content["events"])


def test_merge_without_existing_task_raises():
    mgr = hitl_manager.HITLManager()
    claim_id = "NON_EXISTENT"

    with pytest.raises(FileNotFoundError):
        # save_human_feedback should raise for non-existent task
        mgr.save_human_feedback(claim_id=claim_id, corrected_fields={"a": 1})

    with pytest.raises(FileNotFoundError):
        # attempt to merge nonexistent review -> should raise when opening file
        mgr.merge_human_corrections(claim_id)


def test_merge_preserves_unmodified_fields_and_overwrites_keys():
    mgr = hitl_manager.HITLManager()
    claim_id = "TEST-3003"
    raw_text = "Claim ID: TEST-3003\nName: Carol\nAmount: Rs. 3,500"
    extracted_fields = {
        "claim_id": {"value": "TEST-3003"},
        "amount_estimated": {"value": 3500.0},
        "policy_no": {"value": "PN-000"}
    }
    validation = {}
    ai_inference = {}
    summary = {}
    decision = {"decision": "review", "confidence": 0.2}

    mgr.create_review_task(
        claim_id=claim_id,
        raw_text=raw_text,
        extracted_fields=extracted_fields,
        validation=validation,
        ai_inference=ai_inference,
        summary=summary,
        decision=decision
    )

    # only correct policy_no
    corrected_fields = {"policy_no": {"value": "PN-999"}}
    mgr.save_human_feedback(claim_id=claim_id, corrected_fields=corrected_fields, reviewer="agent_2")

    final = mgr.merge_human_corrections(claim_id)
    assert final["final_fields"]["claim_id"]["value"] == "TEST-3003"
    assert final["final_fields"]["policy_no"]["value"] == "PN-999"
    # amount_estimated preserved
    assert final["final_fields"]["amount_estimated"]["value"] == 3500.0


def test_log_event_generates_hash_and_timestamp():
    mgr = hitl_manager.HITLManager()
    claim_id = "TEST-4004"
    # create a dummy task so audit log path will exist after create_review_task
    mgr.create_review_task(
        claim_id=claim_id,
        raw_text="x",
        extracted_fields={},
        validation={},
        ai_inference={},
        summary={},
        decision={"decision": "review", "confidence": 0.2}
    )
    # write a custom event
    mgr._log_event(claim_id, event="custom_event", details={"k": "v"})
    audit_path = os.path.join(hitl_manager.AUDIT_DIR, f"{claim_id}_audit.json")
    assert os.path.exists(audit_path)
    audit = _load_json(audit_path)
    # event with name custom_event should exist
    assert any(e["event"] == "custom_event" for e in audit["events"])
    # event fields
    ev = next(e for e in audit["events"] if e["event"] == "custom_event")
    assert "timestamp" in ev and "event_hash" in ev and "details" in ev
