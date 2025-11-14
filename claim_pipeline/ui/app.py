import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# claim_pipeline/ui/app.py
"""
Streamlit UI for Claims Pipeline
- Upload Page: upload a claim, run pipeline, view outputs
- HITL Review Page: list pending HITL tasks, edit fields, submit corrections/overrides
- Admin Page: view audit logs and perform maintenance actions
Run:
    streamlit run claim_pipeline/ui/app.py
"""

import os
import json
import tempfile
import streamlit as st
from typing import Dict, Any

# Pipeline & managers (assumes these modules exist per earlier implementation)
from claim_pipeline.pipeline_runner import ClaimsPipeline
from claim_pipeline.hitl.hitl_manager import HITLManager
from claim_pipeline.storage.storage_manager import StorageManager

# Security & helpers
from claim_pipeline.security.auth_manager import get_user_role, has_role, require_role
from claim_pipeline.security.secure_logging import get_logger
from claim_pipeline.security.pii_redactor import redact_dict

logger = get_logger("ui")

# Instantiate singletons
pipeline = ClaimsPipeline()
hitl = HITLManager()
storage = StorageManager()

# UI constants
APP_TITLE = "Intelligent Claims Processing — Demo UI"
STYLES = """
<style>
.reportview-container .main .block-container{padding-top:1rem; padding-left:2rem; padding-right:2rem;}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# ---------------------------
# Utility helpers
# ---------------------------
def write_json_download_button(data: Dict[str, Any], filename: str = "data.json"):
    """Render a download button for JSON data"""
    j = json.dumps(data, indent=2, ensure_ascii=False)
    st.download_button(label="Download JSON", data=j, file_name=filename, mime="application/json")


def show_key_value_editor(prefix: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render editable fields for simple flat mapping (string values expected).
    Returns edited mapping.
    """
    edited = {}
    with st.form(key=f"form_{prefix}", clear_on_submit=False):
        for k, v in fields.items():
            # If value is a dict with {value, confidence} take the .value
            if isinstance(v, dict) and "value" in v:
                default = v.get("value")
            else:
                default = v
            val = st.text_input(label=f"{k}", value=str(default) if default is not None else "", key=f"{prefix}_{k}")
            edited[k] = val.strip() if val != "" else None
        submit = st.form_submit_button("Save Changes Locally")
    return edited


# ---------------------------
# Authentication UI (lightweight)
# ---------------------------
def login_section():
    if "username" not in st.session_state:
        st.session_state["username"] = ""
    st.sidebar.title("User session")
    username = st.sidebar.text_input("Username", value=st.session_state.get("username", ""), help="Enter username (e.g., alice_reviewer)")
    if st.sidebar.button("Set user"):
        st.session_state["username"] = username.strip()
        st.experimental_rerun()

    # Show role
    if st.session_state.get("username"):
        role = get_user_role(st.session_state["username"])
        st.sidebar.markdown(f"**User:** `{st.session_state['username']}`")
        st.sidebar.markdown(f"**Role:** `{role}`")
    else:
        st.sidebar.info("Set a username to continue (use admin/reviewer/auditor/test users from data/users.json)")

    return st.session_state.get("username")


# ---------------------------
# Upload Page
# ---------------------------
def upload_page(username: str):
    st.header("Upload Claim Document")
    st.write("Upload a PDF / image / text file. The pipeline will run end-to-end and display the results.")
    uploaded = st.file_uploader("Choose a claim file", type=["pdf", "png", "jpg", "jpeg", "txt", "eml"], accept_multiple_files=False)

    if uploaded:
        st.info(f"Selected file: {uploaded.name} ({uploaded.size} bytes)")
        # Save to a temporary file and run pipeline
        with st.spinner("Saving file and running pipeline..."):
            tmpdir = tempfile.mkdtemp(prefix="claim_upload_")
            tmp_path = os.path.join(tmpdir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            logger.info("User uploaded file", extra={"username": username, "filename": uploaded.name})
            # Run pipeline
            try:
                result = pipeline.run(tmp_path, uploader=username or "ui_user")
                st.success("Pipeline finished")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                logger.error("Pipeline run failed", extra={"error": str(e)})
                return

        # Display outputs
        st.subheader("Pipeline Output (Live)")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Extracted Text (truncated)**")
            raw_text = result.get("raw_text", "")
            st.text_area("Raw Text", value=raw_text[:20000], height=300)

            st.markdown("**AI Summary**")
            summary = result.get("summary") or {}
            st.json(summary)

        with col2:
            st.markdown("**Decision**")
            decision = result.get("decision", {})
            st.json(decision)

            st.markdown("**Validation**")
            validation = result.get("validation") or {}
            st.json(validation)

            st.markdown("**Storage Paths**")
            st.json(result.get("storage", {}))

        st.markdown("---")
        st.subheader("Extracted Fields (Rule-based + AI)")
        fields = result.get("fields") or {}
        # Show AI-inferred fields nicely (value + confidence)
        pretty = {}
        for k, v in fields.items():
            if isinstance(v, dict) and "value" in v:
                pretty[k] = {"value": v.get("value"), "confidence": v.get("confidence")}
            else:
                pretty[k] = v
        st.json(pretty)

        # allow download of final payload
        write_json_download_button({
            "claim_id": result.get("claim_id"),
            "fields": pretty,
            "validation": validation,
            "summary": summary,
            "decision": decision
        }, filename=f"{result.get('claim_id')}_result.json")

        # If HITL required show CTA
        if result.get("hitl_required"):
            st.warning("This claim requires human review. Please go to the 'HITL Review' page to complete review.")
        else:
            st.success("No human review required for this claim.")

# ---------------------------
# HITL Review Page
# ---------------------------
def hitl_review_page(username: str):
    st.header("Human-in-the-Loop (HITL) - Pending Reviews")

    pending = hitl.get_pending_cases()
    st.info(f"Found {len(pending)} pending task(s) for review")

    if not pending:
        st.write("No pending HITL tasks.")
        return

    for task in pending:
        claim_id = task.get("claim_id")
        with st.expander(f"Task: {claim_id} — Created: {task.get('created_at')} — Status: {task.get('status', 'pending')}"):
            st.markdown("**Why review was triggered:**")
            st.write(task.get("review_reasons", []))

            # Show extracted fields / ai inference / validation
            st.markdown("**AI Inferred Fields**")
            ai_inf = task.get("ai_inference") or task.get("extracted_fields") or {}
            st.json(ai_inf)

            st.markdown("**Validation**")
            st.json(task.get("validation", {}))

            st.markdown("**Summary / Decision**")
            st.json({"summary": task.get("summary"), "decision": task.get("decision")})

            st.markdown("**Editable Fields**")
            # Prepare flat editable mapping (pick ai_inference values if available)
            editable = {}
            for k in ["claim_id","policy_no","claimant_name","phone","email","vehicle","damage","amount_estimated","date"]:
                val = None
                if isinstance(ai_inf, dict) and ai_inf.get(k):
                    v = ai_inf.get(k)
                    if isinstance(v, dict):
                        val = v.get("value")
                    else:
                        val = v
                elif isinstance(task.get("extracted_fields"), dict) and task["extracted_fields"].get(k):
                    val = task["extracted_fields"].get(k)
                editable[k] = val

            edited = show_key_value_editor(prefix=claim_id, fields=editable)

            st.markdown("**Reviewer actions**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"Approve AI output (no change) — {claim_id}"):
                    # Approve: save human feedback with same fields (or empty) and mark as approved
                    try:
                        hitl.save_human_feedback(
                            claim_id=claim_id,
                            corrected_fields=None,
                            corrected_text=None,
                            overridden_decision="approve",
                            comments="Approved by reviewer via UI",
                            reviewer=username or "ui_user"
                        )
                        merged = hitl.merge_human_corrections(claim_id)
                        st.success("Approved and merged. Final result saved.")
                        logger.info("HITL approved", extra={"claim_id": claim_id, "reviewer": username})
                    except Exception as e:
                        st.error(f"Error during approve: {e}")
                        logger.error("HITL approve failed", extra={"error": str(e), "claim_id": claim_id})

            with col2:
                if st.button(f"Submit corrected fields — {claim_id}"):
                    try:
                        # Build corrected_fields from edited mapping (only non-empty)
                        corrected = {k: v for k, v in edited.items() if v is not None and v != ""}
                        if not corrected:
                            st.warning("No corrections entered.")
                        else:
                            hitl.save_human_feedback(
                                claim_id=claim_id,
                                corrected_fields=corrected,
                                corrected_text=None,
                                overridden_decision=None,
                                comments="Corrections submitted via UI",
                                reviewer=username or "ui_user"
                            )
                            merged = hitl.merge_human_corrections(claim_id)
                            st.success("Corrections submitted and merged.")
                            logger.info("HITL corrections submitted", extra={"claim_id": claim_id, "reviewer": username, "corrected_fields": redact_dict(corrected)})
                    except Exception as e:
                        st.error(f"Error submitting corrections: {e}")
                        logger.error("HITL submit failed", extra={"error": str(e), "claim_id": claim_id})

            with col3:
                # Allow override decision
                override = st.selectbox(f"Override decision for {claim_id}", options=["", "approve", "reject", "review"], key=f"ov_{claim_id}")
                if st.button(f"Apply override — {claim_id}"):
                    if not override:
                        st.warning("Select an override value first.")
                    else:
                        try:
                            hitl.save_human_feedback(
                                claim_id=claim_id,
                                corrected_fields=None,
                                corrected_text=None,
                                overridden_decision=override,
                                comments=f"Decision override -> {override} via UI",
                                reviewer=username or "ui_user"
                            )
                            merged = hitl.merge_human_corrections(claim_id)
                            st.success(f"Override applied: {override}")
                            logger.info("HITL override applied", extra={"claim_id": claim_id, "override": override, "reviewer": username})
                        except Exception as e:
                            st.error(f"Error applying override: {e}")
                            logger.error("HITL override failed", extra={"error": str(e), "claim_id": claim_id})

            st.markdown("---")
            # Offer to download task snapshot
            write_json_download_button(task, filename=f"{claim_id}_hitl_task.json")


# ---------------------------
# Admin Page
# ---------------------------
def admin_page(username: str):
    st.header("Admin / Audit")
    st.write("Administrative utilities. Access controlled by `auth_manager` roles. Only admins should use this page.")

    if not has_role(username, "admin"):
        st.error("You must be an admin to use this page.")
        return

    st.subheader("Audit Logs Viewer")
    audit_dir = "data/audit_logs"
    if not os.path.exists(audit_dir):
        st.info("No audit logs directory found.")
        return

    files = sorted([f for f in os.listdir(audit_dir) if f.endswith(".json")])
    selected = st.selectbox("Select audit log", options=[""] + files)
    if selected:
        try:
            with open(os.path.join(audit_dir, selected), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # redact PII for admin preview in logs if desired (but admin can view full)
            st.json(data)
            if st.button("Download Audit JSON"):
                write_json_download_button(data, filename=selected)
        except Exception as e:
            st.error(f"Failed to load audit log: {e}")

    st.subheader("Maintenance")
    if st.button("Run retention archive (archive_old_claims)"):
        from claim_pipeline.security.retention_manager import archive_old_claims
        res = archive_old_claims()
        st.json(res)
        logger.info("Retention archive run by admin", extra={"admin": username})

# ---------------------------
# App main
# ---------------------------
def main():
    st.title(APP_TITLE)
    username = login_section()

    menu = ["Upload", "HITL Review", "Admin"]
    choice = st.sidebar.radio("Navigation", menu)

    if choice == "Upload":
        upload_page(username)
    elif choice == "HITL Review":
        hitl_review_page(username)
    elif choice == "Admin":
        admin_page(username)
    else:
        st.write("Select an option from the sidebar.")


if __name__ == "__main__":
    main()
