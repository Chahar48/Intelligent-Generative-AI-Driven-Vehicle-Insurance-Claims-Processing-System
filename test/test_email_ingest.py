# tests/test_email_ingest.py

import sys
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import pytest

# -------------------------------------------------------------------
# Fake config so import doesn't fail
# -------------------------------------------------------------------
fake_config = MagicMock()
fake_config.RAW_DATA_DIR = "/tmp"
sys.modules["config"] = fake_config
sys.modules["config.__init__"] = fake_config

# -------------------------------------------------------------------
# Fake extract_msg so .msg parsing doesn't fail
# -------------------------------------------------------------------
sys.modules["extract_msg"] = MagicMock()

# -------------------------------------------------------------------
# Import module under test
# -------------------------------------------------------------------
from claim_pipeline.ingestion.email_ingest import (
    extract_email_content,
    _parse_eml,
    _parse_msg,
)


# -------------------------------------------------------------------
# Test: .eml basic plain-text email
# -------------------------------------------------------------------
@patch("builtins.open", new_callable=mock_open, read_data=b"dummy")
@patch("claim_pipeline.ingestion.email_ingest.BytesParser")
def test_parse_eml_plaintext(mock_parser, mock_file, tmp_path):
    dummy_msg = MagicMock()
    dummy_msg.is_multipart.return_value = False
    dummy_msg.get_content_type.return_value = "text/plain"
    dummy_msg.get_content.return_value = "Hello world"
    dummy_msg.get.side_effect = lambda key: {
        "From": "a@test.com",
        "To": "b@test.com",
        "Subject": "Test",
        "Date": "2024-01-01"
    }.get(key)

    mock_parser.return_value.parse.return_value = dummy_msg

    file_path = tmp_path / "email.eml"
    file_path.write_bytes(b"dummy")

    result = extract_email_content(str(file_path))

    assert result["body"] == "Hello world"
    assert result["attachments"] == []
    assert result["metadata"]["from"] == "a@test.com"
    assert result["type"] == "eml"


# -------------------------------------------------------------------
# Test: .eml HTML-only email → converted to text
# -------------------------------------------------------------------
@patch("builtins.open", new_callable=mock_open, read_data=b"dummy")
@patch("claim_pipeline.ingestion.email_ingest.BytesParser")
def test_parse_eml_html_to_text(mock_parser, mock_file, tmp_path):
    dummy_msg = MagicMock()
    dummy_msg.is_multipart.return_value = False
    dummy_msg.get_content_type.return_value = "text/html"
    dummy_msg.get_content.return_value = "<p>Hello <b>HTML</b></p>"
    dummy_msg.get.side_effect = lambda key: {
        "From": "x@test.com",
        "To": "y@test.com",
        "Subject": "HTML",
        "Date": "2024-02-02"
    }.get(key)

    mock_parser.return_value.parse.return_value = dummy_msg

    file_path = tmp_path / "email.eml"
    file_path.write_bytes(b"dummy")

    result = extract_email_content(str(file_path))

    assert "Hello" in result["body"]
    assert "HTML" in result["body"]
    assert result["type"] == "eml"


# -------------------------------------------------------------------
# Test: .eml with attachment
# -------------------------------------------------------------------
@patch("claim_pipeline.ingestion.email_ingest.BytesParser")
def test_parse_eml_with_attachment(mock_parser, tmp_path):
    msg = MagicMock()
    msg.is_multipart.return_value = True

    text_part = MagicMock()
    text_part.get_content_type.return_value = "text/plain"
    text_part.get_content.return_value = "This is body"

    att_part = MagicMock()
    att_part.get_filename.return_value = "file.txt"
    att_part.get_payload.return_value = b"ATTACHMENT"

    msg.walk.return_value = [text_part, att_part]
    msg.get.side_effect = lambda key: {
        "From": "z@test.com",
        "To": "u@test.com",
        "Subject": "Attach",
        "Date": "2024-03-03"
    }.get(key)

    mock_parser.return_value.parse.return_value = msg

    email_path = tmp_path / "email.eml"
    email_path.write_bytes(b"dummy")

    result = extract_email_content(str(email_path))

    assert result["body"] == "This is body"
    assert len(result["attachments"]) == 1
    assert Path(result["attachments"][0]).exists()


# -------------------------------------------------------------------
# Test: .msg basic parsing
# -------------------------------------------------------------------
@patch("extract_msg.Message")
def test_parse_msg_basic(mock_msg_class, tmp_path):
    msg_instance = MagicMock()
    msg_instance.sender = "x@test.com"
    msg_instance.to = "y@test.com"
    msg_instance.subject = "Hello"
    msg_instance.date = "2024-01-01"
    msg_instance.body = "Message body"
    msg_instance.htmlBody = None
    msg_instance.attachments = []

    mock_msg_class.return_value = msg_instance

    email_path = tmp_path / "email.msg"
    email_path.write_bytes(b"dummy")

    result = extract_email_content(str(email_path))

    assert result["body"] == "Message body"
    assert result["metadata"]["from"] == "x@test.com"
    assert result["type"] == "msg"


# -------------------------------------------------------------------
# Test: .msg with HTML-only body
# -------------------------------------------------------------------
@patch("extract_msg.Message")
def test_parse_msg_html_only(mock_msg_class, tmp_path):
    msg_instance = MagicMock()
    msg_instance.body = ""
    msg_instance.htmlBody = "<p>Hi <b>there</b></p>"
    msg_instance.attachments = []
    msg_instance.sender = "a@test.com"
    msg_instance.to = "b@test.com"
    msg_instance.subject = "HTML MSG"
    msg_instance.date = "2024-04-04"

    mock_msg_class.return_value = msg_instance

    email_path = tmp_path / "email.msg"
    email_path.write_bytes(b"dummy")

    result = extract_email_content(str(email_path))

    assert "Hi" in result["body"]
    assert "there" in result["body"]
    assert result["type"] == "msg"


# -------------------------------------------------------------------
# Test: .msg with attachments
# -------------------------------------------------------------------
@patch("extract_msg.Message")
def test_parse_msg_with_attachments(mock_msg_class, tmp_path):
    attachment = MagicMock()
    attachment.longFilename = "doc.pdf"
    attachment.shortFilename = None
    attachment.data = b"PDF-DATA"

    msg_instance = MagicMock()
    msg_instance.body = "Hello"
    msg_instance.htmlBody = None
    msg_instance.attachments = [attachment]
    msg_instance.sender = "s@test.com"
    msg_instance.to = "t@test.com"
    msg_instance.subject = "Files"
    msg_instance.date = "2024-05-05"

    mock_msg_class.return_value = msg_instance

    email_path = tmp_path / "email.msg"
    email_path.write_bytes(b"dummy")

    result = extract_email_content(str(email_path))

    assert result["body"] == "Hello"
    assert len(result["attachments"]) == 1
    assert Path(result["attachments"][0]).exists()


# -------------------------------------------------------------------
# Test: unsupported file type
# -------------------------------------------------------------------
def test_unsupported_email_type(tmp_path):
    file_path = tmp_path / "lol.xyz"
    file_path.write_text("dummy")

    result = extract_email_content(str(file_path))

    assert result == {}
