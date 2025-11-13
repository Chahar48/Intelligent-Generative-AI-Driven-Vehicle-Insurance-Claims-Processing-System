# claim_pipeline/data_ingestion/email_ingest.py

import os
import logging
import email
import base64
from email import policy
from email.parser import BytesParser
from typing import Dict, List, Any
from bs4 import BeautifulSoup
from config import RAW_DATA_DIR

# For .msg file support
try:
    import extract_msg   # lightweight Outlook .msg parser
except ImportError:
    extract_msg = None


# ----------------------------------------------------
# Logging
# ----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ----------------------------------------------------
# Helper: clean HTML → plain text
# ----------------------------------------------------
def html_to_text(html_content: str) -> str:
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n")
    except Exception:
        return ""


# ----------------------------------------------------
# Helper: detect presence of email-like text inside attachments
# ----------------------------------------------------
def detect_email_content(text: str) -> bool:
    email_indicators = ["From:", "To:", "Subject:", "Sent:", "Forwarded message"]
    txt = text.lower()
    return any(ind.lower() in txt for ind in email_indicators)


# ----------------------------------------------------
# Handle .eml files
# ----------------------------------------------------
def _parse_eml(email_path: str, claim_folder: str) -> Dict[str, Any]:
    """
    Extracts body, metadata, and attachments from .eml email.
    """
    body_text = ""
    attachments = []
    metadata = {"from": None, "to": None, "subject": None, "date": None}

    try:
        with open(email_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        metadata["from"] = msg.get("From")
        metadata["to"] = msg.get("To")
        metadata["subject"] = msg.get("Subject")
        metadata["date"] = msg.get("Date")

        # Walk through parts
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()

                # Prefer plain-text
                if content_type == "text/plain":
                    body_text += part.get_content()

                # If only HTML exists
                elif content_type == "text/html" and not body_text.strip():
                    html_content = part.get_content()
                    body_text = html_to_text(html_content)

                # Attachment
                elif part.get_filename():
                    filename = part.get_filename()
                    safe_name = filename.replace("..", "").replace("/", "_")
                    att_path = os.path.join(claim_folder, safe_name)

                    with open(att_path, "wb") as out:
                        out.write(part.get_payload(decode=True))

                    attachments.append(att_path)
                    logger.info(f"[EmailIngest] Saved attachment: {att_path}")

        else:
            # Single-part email
            content_type = msg.get_content_type()
            content = msg.get_content()

            if content_type == "text/plain":
                body_text = content
            elif content_type == "text/html":
                body_text = html_to_text(content)

    except Exception as e:
        logger.error(f"[EmailIngest] Error parsing .eml file {email_path}: {e}")

    # Detect embedded email signatures or forwarded email content
    embedded_email = detect_email_content(body_text)

    return {
        "body": body_text,
        "attachments": attachments,
        "metadata": metadata,
        "contains_embedded_email": embedded_email,
        "type": "eml"
    }


# ----------------------------------------------------
# Handle .msg files
# ----------------------------------------------------
def _parse_msg(email_path: str, claim_folder: str) -> Dict[str, Any]:
    """
    Extracts body, metadata, and attachments from .msg Outlook emails.
    """
    if extract_msg is None:
        logger.error("[EmailIngest] Cannot parse .msg because 'extract_msg' is not installed.")
        return {
            "body": "",
            "attachments": [],
            "metadata": {},
            "contains_embedded_email": False,
            "type": "msg"
        }

    try:
        msg = extract_msg.Message(email_path)

        metadata = {
            "from": msg.sender,
            "to": msg.to,
            "subject": msg.subject,
            "date": msg.date
        }

        # Body: .msg usually has plain text + HTML
        body_text = msg.body or ""
        if not body_text.strip() and msg.htmlBody:
            body_text = html_to_text(msg.htmlBody)

        # Save attachments
        attachments = []
        for att in msg.attachments:
            filename = att.longFilename or att.shortFilename
            safe_name = filename.replace("..", "").replace("/", "_")
            att_path = os.path.join(claim_folder, safe_name)

            with open(att_path, "wb") as out:
                out.write(att.data)

            attachments.append(att_path)
            logger.info(f"[EmailIngest] Saved .msg attachment: {att_path}")

        embedded_email = detect_email_content(body_text)

        return {
            "body": body_text,
            "attachments": attachments,
            "metadata": metadata,
            "contains_embedded_email": embedded_email,
            "type": "msg"
        }

    except Exception as e:
        logger.error(f"[EmailIngest] Error parsing .msg file {email_path}: {e}")
        return {
            "body": "",
            "attachments": [],
            "metadata": {},
            "contains_embedded_email": False,
            "type": "msg"
        }


# ----------------------------------------------------
# Public API
# ----------------------------------------------------
def extract_email_content(email_path: str) -> Dict[str, Any]:
    """
    Return structured email extraction:
    {
        "body": "<string>",
        "attachments": ["paths"],
        "metadata": {"from":..., "to":..., "subject":..., "date":...},
        "contains_embedded_email": True/False,
        "type": "eml|msg"
    }
    """
    claim_folder = os.path.dirname(email_path)

    if email_path.lower().endswith(".eml"):
        return _parse_eml(email_path, claim_folder)

    elif email_path.lower().endswith(".msg"):
        return _parse_msg(email_path, claim_folder)

    else:
        logger.error(f"[EmailIngest] Unsupported email file type: {email_path}")
        return {}