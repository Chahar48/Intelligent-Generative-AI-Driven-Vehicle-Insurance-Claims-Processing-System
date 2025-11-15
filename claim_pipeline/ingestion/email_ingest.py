# claim_pipeline/ingestion/email_ingest.py

import os
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup

# Optional .msg support (Outlook emails)
try:
    import extract_msg
except ImportError:
    extract_msg = None


# -------------------------------------------------
# Helper: Convert HTML → plain text
# -------------------------------------------------
def html_to_text(html_content: str) -> str:
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n")
    except Exception:
        return ""


# -------------------------------------------------
# Parse .eml files (simple)
# -------------------------------------------------
def _parse_eml(email_path: str, claim_folder: str):
    body = ""
    attachments = []
    metadata = {}

    try:
        with open(email_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        metadata = {
            "from": msg.get("From"),
            "to": msg.get("To"),
            "subject": msg.get("Subject"),
            "date": msg.get("Date"),
        }

        # If email has multiple parts
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()

                # Prefer plain text
                if ctype == "text/plain":
                    body = part.get_content()

                # Fallback to HTML
                elif ctype == "text/html" and not body.strip():
                    body = html_to_text(part.get_content())

                # Save attachments
                elif part.get_filename():
                    filename = part.get_filename()
                    safe_name = filename.replace("..", "").replace("/", "_")
                    save_path = os.path.join(claim_folder, safe_name)

                    with open(save_path, "wb") as out:
                        out.write(part.get_payload(decode=True))

                    attachments.append(save_path)

        else:
            # Single-part email
            ctype = msg.get_content_type()
            content = msg.get_content()

            if ctype == "text/plain":
                body = content
            elif ctype == "text/html":
                body = html_to_text(content)

    except Exception:
        pass  # keep it simple for beginners

    return {
        "type": "eml",
        "body": body,
        "metadata": metadata,
        "attachments": attachments
    }


# -------------------------------------------------
# Parse .msg files (simple)
# -------------------------------------------------
def _parse_msg(email_path: str, claim_folder: str):
    if extract_msg is None:
        return {
            "type": "msg",
            "body": "",
            "metadata": {},
            "attachments": []
        }

    try:
        msg = extract_msg.Message(email_path)

        metadata = {
            "from": msg.sender,
            "to": msg.to,
            "subject": msg.subject,
            "date": msg.date,
        }

        # Simple body extraction
        body = msg.body or ""
        if not body.strip() and msg.htmlBody:
            body = html_to_text(msg.htmlBody)

        # Save attachments
        attachments = []
        for att in msg.attachments:
            filename = att.longFilename or att.shortFilename or "attachment"
            safe_name = filename.replace("..", "").replace("/", "_")
            save_path = os.path.join(claim_folder, safe_name)

            with open(save_path, "wb") as out:
                out.write(att.data)

            attachments.append(save_path)

        return {
            "type": "msg",
            "body": body,
            "metadata": metadata,
            "attachments": attachments
        }

    except Exception:
        return {
            "type": "msg",
            "body": "",
            "metadata": {},
            "attachments": []
        }


# -------------------------------------------------
# Public Function
# -------------------------------------------------
def extract_email_content(email_path: str):
    """
    Always returns:
    {
        "type": "eml" | "msg",
        "body": "...",
        "metadata": {...},
        "attachments": [...]
    }
    """
    claim_folder = os.path.dirname(email_path)

    if email_path.lower().endswith(".eml"):
        return _parse_eml(email_path, claim_folder)

    if email_path.lower().endswith(".msg"):
        return _parse_msg(email_path, claim_folder)

    # Unsupported
    return {
        "type": "unknown",
        "body": "",
        "metadata": {},
        "attachments": []
    }

