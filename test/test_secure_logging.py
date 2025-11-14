import os
import json
import logging
import shutil
import pytest

from claim_pipeline.security.secure_logging import get_logger, LOG_DIR


# -----------------------------------------------------
# Setup/Cleanup for log environment
# -----------------------------------------------------
def setup_module(module):
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def teardown_module(module):
    shutil.rmtree(LOG_DIR, ignore_errors=True)


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def read_log_file(filename="claims.log"):
    path = os.path.join(LOG_DIR, filename)
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
#                        BASIC TESTS
# ============================================================
def test_logger_basic_message_redaction():
    logger = get_logger("test_basic", "test_basic.log")

    logger.info("User email is john.doe@gmail.com")

    data = read_log_file("test_basic.log")

    assert "gmail.com" in data                      # domain allowed
    assert "john.doe@" not in data                  # username must be masked
    assert "joh" in data                            # first few chars kept
    assert "***" in data                            # star masking visible


def test_logger_extra_dict_redaction():
    logger = get_logger("test_dict", "test_dict.log")

    payload = {
        "email": "john.doe@gmail.com",
        "phone": "+91-9876543210",
        "claim_id": "CLM-998877"
    }

    logger.info("Test event", extra={"payload": payload})

    data = read_log_file("test_dict.log")

    # Check redaction for each type
    assert "joh" in data and "***" in data           # email masked
    assert "98******10" in data                      # phone masked
    assert "CLM-" in data and "*" in data            # claim ID masked


def test_logger_extra_list_redaction():
    logger = get_logger("test_list", "test_list.log")

    payload = [
        "john.doe@gmail.com",
        "+91-9876543210",
        "CLM-223344"
    ]

    logger.info("Event with list", extra={"payload": payload})

    data = read_log_file("test_list.log")

    assert "gmail.com" in data
    assert "joh" in data and "*" in data
    assert "98******10" in data
    assert "CLM-" in data and "*" in data


def test_logger_file_output():
    logger = get_logger("test_file", "test_file.log")

    logger.info("Phone 99999-88888")
    data = read_log_file("test_file.log")

    assert "***" in data or "****" in data
    assert "9999" not in data[-10:]  # digits removed from masked output


def test_logger_reuse_does_not_add_handlers():
    logger1 = get_logger("reuse_test", "reuse.log")
    handler_count_1 = len(logger1.handlers)

    logger2 = get_logger("reuse_test", "reuse.log")
    handler_count_2 = len(logger2.handlers)

    assert logger1 is logger2
    assert handler_count_1 == handler_count_2            # no duplicate handlers


# ============================================================
#                EDGE CASES / ROBUSTNESS TESTS
# ============================================================
def test_logger_handles_non_serializable_objects():
    logger = get_logger("test_non_serializable", "test_non_serializable.log")

    class BadObj:
        pass

    logger.info("Bad object test", extra={"obj": BadObj()})

    data = read_log_file("test_non_serializable.log")

    # Should not crash; log should contain placeholder or safe repr
    assert "extras" in data
    assert "BadObj" in data or "<redaction_error>" in data


def test_formatter_does_not_crash_on_bad_input():
    logger = get_logger("test_bad_input", "test_bad_input.log")

    # Force weird/unexpected log record values
    logger.info(None)
    logger.info(12345)
    logger.info({"a": 1})  # dict message

    data = read_log_file("test_bad_input.log")

    assert "extras" in data or "{}" in data  # Should not crash formatter