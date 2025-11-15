# claim_pipeline/ui/app.py
import sys
import os
import json
import tempfile
import streamlit as st
from typing import Dict, Any, List, Optional

# Add project root to path when running directly (keeps old behavior)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Pipeline & managers (assumes these modules exist per earlier implementation)
from claim_pipeline.pipeline_runner import ClaimsPipeline
from claim_pipeline.hitl.hitl_manager import HITLManager
from claim_pipeline.storage.storage_manager import StorageManager

# Security & helpers
#from claim_pipeline.security.auth_manager import get_user_role, has_role, require_role
from claim_pipeline.security.pii_redactor import redact_dict

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


def _field_value(v):
    """
    Helper to extract a primitive value from a field which might be:
    - primitive (str/int/float)
    - dict with .get('value')
    """
    if isinstance(v, dict):
        return v.get("value")
    return v


def show_key_value_editor(prefix: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render editable fields for simple flat mapping (string values expected).
    Returns edited mapping.
    """
    edited = {}
    with st.form(key=f"form_{prefix}", clear_on_submit=False):
        for k, v in fields.items():
            default = _field_value(v)
            val = st.text_input(label=f"{k}", value=str(default) if default is not None else "", key=f"{prefix}_{k}")
            edited[k] = val.strip() if val != "" else None
        st.form_submit_button("Save Changes Locally")
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

    # # Show role
    # if st.session_state.get("username"):
    #     role = get_user_role(st.session_state["username"])
    #     st.sidebar.markdown(f"**User:** `{st.session_state['username']}`")
    #     st.sidebar.markdown(f"**Role:** `{role}`")
    # else:
    #     st.sidebar.info("Set a username to continue (use admin/reviewer/auditor/test users from data/users.json)")

    # return st.session_state.get("username")


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
            try:
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
            except Exception as e:
                st.error(f"Failed saving uploaded file: {e}")
                return

            # Run pipeline defensively
            try:
                result = pipeline.run(tmp_path, uploader=username or "ui_user")
                st.success("Pipeline finished")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                return

        # Display outputs
        st.subheader("Pipeline Output (Live)")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Extracted Text (truncated)**")
            raw_text = result.get("raw_text", "") or ""
            st.text_area("Raw Text", value=raw_text[:20000], height=300)

            st.markdown("**AI Summary**")
            summary = result.get("summary") or {}
            st.json(summary)

        with col2:
            st.markdown("**Decision**")
            decision = result.get("decision", {}) or {}
            st.json(decision)

            st.markdown("**Validation**")
            validation = result.get("validation") or {}
            st.json(validation)

            st.markdown("**Storage Paths**")
            st.json(result.get("storage", {}) or {})

        st.markdown("---")
        st.subheader("Extracted Fields (Rule-based + AI)")

        # 'fields' kept in pipeline as ai_fields (dict). Normalize display.
        fields = result.get("fields") or {}
        pretty = {}
        if isinstance(fields, dict):
            for k, v in fields.items():
                if isinstance(v, dict):
                    pretty[k] = {
                        "value": v.get("value"),
                        "confidence": v.get("confidence") if isinstance(v.get("confidence"), (int, float)) else v.get("score")
                    }
                else:
                    pretty[k] = v
        else:
            pretty = fields

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
                    val = _field_value(v)
                elif isinstance(task.get("extracted_fields"), dict):
                    # extracted_fields raw_lines + fields shape; try to get fields[k]
                    ef = task.get("extracted_fields") or {}
                    fields_map = ef.get("fields") if isinstance(ef, dict) else None
                    if isinstance(fields_map, dict) and fields_map.get(k):
                        val = _field_value(fields_map.get(k))
                editable[k] = val

            edited = show_key_value_editor(prefix=claim_id, fields=editable)

            st.markdown("**Reviewer actions**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"Approve AI output (no change) — {claim_id}"):
                    try:
                        hitl.save_human_feedback(
                            claim_id=claim_id,
                            corrected_fields=None,
                            corrected_text=None,
                            overridden_decision="approve",
                            comments="Approved by reviewer via UI",
                            reviewer=username or "ui_user"
                        )
                        _ = hitl.merge_human_corrections(claim_id)
                        st.success("Approved and merged. Final result saved.")
                    except Exception as e:
                        st.error(f"Error during approve: {e}")

            with col2:
                if st.button(f"Submit corrected fields — {claim_id}"):
                    try:
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
                            _ = hitl.merge_human_corrections(claim_id)
                            st.success("Corrections submitted and merged.")
                    except Exception as e:
                        st.error(f"Error submitting corrections: {e}")

            with col3:
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
                            _ = hitl.merge_human_corrections(claim_id)
                            st.success(f"Override applied: {override}")
                        except Exception as e:
                            st.error(f"Error applying override: {e}")

            st.markdown("---")
            # Offer to download task snapshot (PII redacted for convenience)
            try:
                snapshot = redact_dict(task)
            except Exception:
                snapshot = task
            write_json_download_button(snapshot, filename=f"{claim_id}_hitl_task.json")


#---------------------------
#Admin Page
#---------------------------
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
            st.json(data)
            if st.button("Download Audit JSON"):
                write_json_download_button(data, filename=selected)
        except Exception as e:
            st.error(f"Failed to load audit log: {e}")

    st.subheader("Maintenance")
    if st.button("Run retention archive (archive_old_claims)"):
        try:
            from claim_pipeline.security.retention_manager import archive_old_claims
            res = archive_old_claims()
            st.json(res)
        except Exception as e:
            st.error(f"Retention archive failed: {e}")


# ---------------------------
# App main
# ---------------------------
def main():
    st.title(APP_TITLE)
    username = login_section()

    menu = ["Upload", "HITL Review"]
    choice = st.sidebar.radio("Navigation", menu)

    if choice == "Upload":
        upload_page(username)
    elif choice == "HITL Review":
        hitl_review_page(username)
    else:
        st.write("Select an option from the sidebar.")


if __name__ == "__main__":
    main()



