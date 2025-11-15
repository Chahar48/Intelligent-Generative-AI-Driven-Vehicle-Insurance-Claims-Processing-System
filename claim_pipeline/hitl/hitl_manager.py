# claim_pipeline/hitl/hitl_manager.py

"""
Clean HITL Manager (logging-free)
---------------------------------
Handles:
- Creating review tasks
- Loading pending reviews
- Saving human corrections
- Merging corrections into final output
- JSON audit trail (not Python logging)
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, List, Optional

REVIEW_DIR = "data/reviews"
AUDIT_DIR = "data/audit_logs"
FINAL_DIR = "data/final"

os.makedirs(REVIEW_DIR, exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)


# =====================================================
# Helpers (pure, logging-free)
# =====================================================

def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _save_json(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =====================================================
# HITL Manager (fully safe, logging-removed)
# =====================================================

class HITLManager:

    # -------------------------------------------------
    # 1) Create a review task
    # -------------------------------------------------
    def create_review_task(
        self,
        claim_id: str,
        raw_text: str,
        extracted_fields: Dict[str, Any],
        validation: Dict[str, Any],
        ai_inference: Dict[str, Any],
        summary: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:

        task = {
            "claim_id": claim_id,
            "created_at": _now_ts(),
            "raw_text": raw_text,

            "extracted_fields": extracted_fields,
            "validation": validation,
            "ai_inference": ai_inference,
            "summary": summary,
            "decision": decision,

            "review_reasons": self._derive_reasons(validation, ai_inference, decision),
            "status": "pending"
        }

        task_path = os.path.join(REVIEW_DIR, f"{claim_id}.json")
        _save_json(task_path, task)

        self._log_event(claim_id, "review_task_created", {"reasons": task["review_reasons"]})

        return task

    def _derive_reasons(self, validation, ai_inference, decision):
        reasons = []

        if validation.get("issues"):
            reasons.append("validation_issues")

        if validation.get("missing_required"):
            reasons.append("missing_required_fields")

        # Safe confidence evaluation
        low_conf = []
        for f, v in ai_inference.items():
            if isinstance(v, dict):
                conf = v.get("confidence", 1)
                try:
                    conf_f = float(conf)
                except:
                    conf_f = 1.0
                if conf_f < 0.5:
                    low_conf.append(f)

        if low_conf:
            reasons.append(f"low_confidence_fields: {low_conf}")

        dec = decision.get("decision")
        if dec in ("review", "escalate"):
            reasons.append("decision_requires_human_review")

        try:
            dec_conf = float(decision.get("confidence", 1))
        except:
            dec_conf = 1.0

        if dec_conf < 0.8:
            reasons.append("low_decision_confidence")

        return reasons or ["unspecified_review_reason"]

    # -------------------------------------------------
    # 2) Load all pending cases
    # -------------------------------------------------
    def get_pending_cases(self) -> List[Dict[str, Any]]:
        out = []
        for fn in os.listdir(REVIEW_DIR):
            if fn.endswith(".json"):
                data = _load_json(os.path.join(REVIEW_DIR, fn))
                if data.get("status") == "pending":
                    out.append(data)
        return out

    # -------------------------------------------------
    # 3) Save human corrections
    # -------------------------------------------------
    def save_human_feedback(
        self,
        claim_id: str,
        corrected_fields: Optional[Dict[str, Any]] = None,
        corrected_text: Optional[str] = None,
        overridden_decision: Optional[str] = None,
        comments: Optional[str] = None,
        reviewer: str = "human_reviewer"
    ):

        path = os.path.join(REVIEW_DIR, f"{claim_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No review task found for claim {claim_id}")

        task = _load_json(path)

        task["status"] = "review_completed"
        task["reviewed_at"] = _now_ts()
        task["reviewer"] = reviewer

        if corrected_fields:
            task["human_corrected_fields"] = corrected_fields

        if corrected_text:
            task["human_corrected_text"] = corrected_text

        if overridden_decision:
            task["human_overridden_decision"] = overridden_decision

        if comments:
            task["reviewer_comments"] = comments

        _save_json(path, task)

        self._log_event(
            claim_id,
            "human_review_submitted",
            {
                "corrected_fields": corrected_fields,
                "corrected_text_present": bool(corrected_text),
                "overridden_decision": overridden_decision,
                "comments": comments
            }
        )

        return task

    # -------------------------------------------------
    # 4) Merge corrections into final result
    # -------------------------------------------------
    def merge_human_corrections(self, claim_id: str) -> Dict[str, Any]:

        path = os.path.join(REVIEW_DIR, f"{claim_id}.json")
        task = _load_json(path)

        final_fields = self._merge_fields(
            task.get("extracted_fields", {}),
            task.get("human_corrected_fields", {})
        )

        final_result = {
            "claim_id": claim_id,
            "finalized_at": _now_ts(),
            "raw_text": task.get("human_corrected_text") or task.get("raw_text"),
            "final_fields": final_fields,
            "final_decision": task.get("human_overridden_decision", task["decision"]["decision"]),
            "review_comments": task.get("reviewer_comments"),
            "reviewer": task.get("reviewer")
        }

        final_path = os.path.join(FINAL_DIR, f"{claim_id}_final.json")
        _save_json(final_path, final_result)

        self._log_event(claim_id, "final_output_generated", {"final_decision": final_result["final_decision"]})

        return final_result

    def _merge_fields(self, extracted_fields, corrected_fields):
        merged = extracted_fields.copy()
        for k, v in corrected_fields.items():
            merged[k] = v
        return merged

    # -------------------------------------------------
    # 5) JSON audit log (not Python logging)
    # -------------------------------------------------
    def _log_event(self, claim_id: str, event: str, details: Dict[str, Any]):
        log_path = os.path.join(AUDIT_DIR, f"{claim_id}_audit.json")

        if os.path.exists(log_path):
            audit = _load_json(log_path)
        else:
            audit = {"claim_id": claim_id, "events": []}

        entry = {
            "timestamp": _now_ts(),
            "event": event,
            "details": details,
            "event_hash": _sha256(event + str(details) + _now_ts())
        }

        audit["events"].append(entry)
        _save_json(log_path, audit)




############################################################################################################

# claim_pipeline/hitl/hitl_manager.py

"""
Human-in-the-Loop (HITL) Manager for Claims Pipeline
----------------------------------------------------

Responsibilities:
✔ Create review tasks when pipeline flags low-confidence outputs
✔ Load pending human-review cases
✔ Save human corrections (OCR, fields, decisions)
✔ Merge human corrections back into pipeline
✔ Maintain complete audit trail (JSON logs)
✔ Ensure final corrected outputs are persisted

This module works fully with JSON-based storage (no DB).
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, List, Optional

# Storage directories (JSON-based)
REVIEW_DIR = "data/reviews"
AUDIT_DIR = "data/audit_logs"
FINAL_DIR = "data/final"

os.makedirs(REVIEW_DIR, exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)


# -----------------------------------------------------------
# Utility Helpers
# -----------------------------------------------------------

def _now_ts():
    """Human-readable timestamp."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _save_json(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -----------------------------------------------------------
# HITL: Core Manager
# -----------------------------------------------------------

class HITLManager:

    # -------------------------------------------------------
    # 1) Create Review Task
    # -------------------------------------------------------
    def create_review_task(
        self,
        claim_id: str,
        raw_text: str,
        extracted_fields: Dict[str, Any],
        validation: Dict[str, Any],
        ai_inference: Dict[str, Any],
        summary: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Creates a review task when pipeline detects low-confidence or issues.
        Stores a JSON snapshot for human reviewers.
        """

        task = {
            "claim_id": claim_id,
            "created_at": _now_ts(),

            # raw inputs
            "raw_text": raw_text,

            # extracted fields
            "extracted_fields": extracted_fields,
            "validation": validation,
            "ai_inference": ai_inference,
            "summary": summary,
            "decision": decision,

            # why human action is needed
            "review_reasons": self._derive_reasons(validation, ai_inference, decision),

            "status": "pending"
        }

        path = os.path.join(REVIEW_DIR, f"{claim_id}.json")
        _save_json(path, task)

        self._log_event(
            claim_id,
            event="review_task_created",
            details={"reasons": task["review_reasons"]}
        )

        return task

    def _derive_reasons(self, validation, ai_inference, decision):
        """Extract reasons explaining WHY human review was triggered."""
        reasons = []

        # validation issues
        if validation.get("issues"):
            reasons.append("validation_issues")

        if validation.get("missing_required"):
            reasons.append("missing_required_fields")

        # ai confidence
        low_conf = [
            f for f, v in ai_inference.items()
            if isinstance(v, dict) and v.get("confidence", 1) < 0.50
        ]
        if low_conf:
            reasons.append(f"low_confidence_fields: {low_conf}")

        # decision
        if decision.get("decision") in ("review", "escalate"):
            reasons.append("decision_requires_human_review")

        if decision.get("confidence", 1) < 0.8:
            reasons.append("low_decision_confidence")

        return reasons or ["unspecified_review_reason"]

    # -------------------------------------------------------
    # 2) Load Pending Cases
    # -------------------------------------------------------
    def get_pending_cases(self) -> List[Dict[str, Any]]:
        """
        Returns all pending review tasks.
        """
        tasks = []
        for fn in os.listdir(REVIEW_DIR):
            if fn.endswith(".json"):
                data = _load_json(os.path.join(REVIEW_DIR, fn))
                if data.get("status") == "pending":
                    tasks.append(data)
        return tasks

    # -------------------------------------------------------
    # 3) Save Human Corrections
    # -------------------------------------------------------
    def save_human_feedback(
        self,
        claim_id: str,
        corrected_fields: Optional[Dict[str, Any]] = None,
        corrected_text: Optional[str] = None,
        overridden_decision: Optional[str] = None,
        comments: Optional[str] = None,
        reviewer: str = "human_reviewer"
    ):
        """
        Saves human corrections into the review task JSON.
        """

        path = os.path.join(REVIEW_DIR, f"{claim_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No review task found for claim {claim_id}")

        task = _load_json(path)

        task["status"] = "review_completed"
        task["reviewed_at"] = _now_ts()
        task["reviewer"] = reviewer

        if corrected_fields:
            task["human_corrected_fields"] = corrected_fields

        if corrected_text:
            task["human_corrected_text"] = corrected_text

        if overridden_decision:
            task["human_overridden_decision"] = overridden_decision

        if comments:
            task["reviewer_comments"] = comments

        _save_json(path, task)

        self._log_event(
            claim_id,
            event="human_review_submitted",
            details={
                "corrected_fields": corrected_fields,
                "corrected_text": True if corrected_text else False,
                "overridden_decision": overridden_decision,
                "comments": comments
            }
        )

        return task

    # -------------------------------------------------------
    # 4) Merge Human Corrections Back Into Pipeline
    # -------------------------------------------------------
    def merge_human_corrections(self, claim_id: str) -> Dict[str, Any]:
        """
        Takes human-corrected values and generates a final pipeline result.
        """

        path = os.path.join(REVIEW_DIR, f"{claim_id}.json")
        task = _load_json(path)

        final = {
            "claim_id": claim_id,
            "finalized_at": _now_ts(),
            "raw_text": task.get("human_corrected_text") or task.get("raw_text"),

            # Merge fields
            "final_fields": self._merge_fields(
                task.get("extracted_fields", {}),
                task.get("human_corrected_fields", {})
            ),

            # Decision override
            "final_decision": task.get("human_overridden_decision", task["decision"]["decision"]),

            "review_comments": task.get("reviewer_comments"),
            "reviewer": task.get("reviewer"),
        }

        final_path = os.path.join(FINAL_DIR, f"{claim_id}_final.json")
        _save_json(final_path, final)

        self._log_event(
            claim_id,
            event="final_output_generated",
            details={"final_decision": final["final_decision"]}
        )

        return final

    def _merge_fields(self, extracted_fields, corrected_fields):
        """
        Replaces AI/Rule-based extracted fields with human corrections.
        """
        merged = extracted_fields.copy()
        for key, val in corrected_fields.items():
            merged[key] = val
        return merged

    # -------------------------------------------------------
    # 5) Audit Logging
    # -------------------------------------------------------
    def _log_event(self, claim_id: str, event: str, details: Dict[str, Any]):
        """
        Stores a timestamped audit event for traceability.
        """
        log_path = os.path.join(AUDIT_DIR, f"{claim_id}_audit.json")

        if os.path.exists(log_path):
            audit = _load_json(log_path)
        else:
            audit = {"claim_id": claim_id, "events": []}

        audit["events"].append({
            "timestamp": _now_ts(),
            "event": event,
            "details": details,
            "event_hash": _sha256(f"{event}{details}{_now_ts()}")
        })

        _save_json(log_path, audit)

