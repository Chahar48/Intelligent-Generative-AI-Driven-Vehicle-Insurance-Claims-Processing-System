import logging
import os
import json
from logging import Logger
from typing import Any
from .pii_redactor import redact_dict, redact_text

class RedactingFormatter(logging.Formatter):
    INTERNAL_KEYS = {
        "name", "msg", "args", "levelname", "levelno",
        "pathname", "filename", "module", "exc_info",
        "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process"
    }

    def format(self, record: logging.LogRecord) -> str:
        # Redact message text
        msg = record.getMessage()
        try:
            msg = redact_text(msg)
        except Exception:
            pass

        # Get sanitized extras (only user-passed extras)
        sanitized = {}
        for k, v in record.__dict__.items():
            if k in self.INTERNAL_KEYS:
                continue

            try:
                if isinstance(v, dict):
                    sanitized[k] = redact_dict(v)
                elif isinstance(v, list):
                    sanitized[k] = redact_dict(v)
                elif isinstance(v, str):
                    sanitized[k] = redact_text(v)
                else:
                    sanitized[k] = v
            except Exception:
                sanitized[k] = "<redaction_error>"

        # Format base log entry
        base = super().format(record)

        if sanitized:
            return f"{base} | extras={json.dumps(sanitized, default=str, ensure_ascii=False)}"

        return base


# # claim_pipeline/security/secure_logging.py
# """
# Secure logging wrapper that redacts PII before logging.

# Usage:
#     from claim_pipeline.security.secure_logging import get_logger
#     logger = get_logger("claims")
#     logger.info("Processing claim", extra={"claim": claim_dict})  # claim_dict will be redacted
# """

# import logging
# import os
# import json
# from logging import Logger
# from typing import Any
# from .pii_redactor import redact_dict, redact_text

# LOG_DIR = "logs"
# os.makedirs(LOG_DIR, exist_ok=True)

# class RedactingFormatter(logging.Formatter):
#     def format(self, record: logging.LogRecord) -> str:
#         # If message is a dict-like in record.args or record.__dict__ extras,
#         # attempt to redact sensitive info.
#         msg = record.getMessage()
#         try:
#             # redact plain text message
#             msg = redact_text(msg)
#         except Exception:
#             pass

#         # redact extra fields if present (e.g., record.__dict__['extra'])
#         # Build a sanitized dict for structured logging
#         sanitized = {}
#         for k, v in record.__dict__.items():
#             if k in ("msg", "args", "levelno", "levelname", "name", "msg"):
#                 continue
#             try:
#                 sanitized[k] = redact_dict(v) if isinstance(v, (dict, list)) else (redact_text(v) if isinstance(v, str) else v)
#             except Exception:
#                 sanitized[k] = "<redaction_error>"

#         base = super().format(record)
#         if sanitized:
#             return f"{base} | extras={json.dumps(sanitized, default=str, ensure_ascii=False)}"
#         return base

# def get_logger(name: str = "claims", log_filename: str = "claims.log") -> Logger:
#     logger = logging.getLogger(name)
#     if logger.handlers:
#         return logger

#     logger.setLevel(logging.INFO)

#     # Console handler
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     ch_formatter = RedactingFormatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
#     ch.setFormatter(ch_formatter)
#     logger.addHandler(ch)

#     # File handler
#     fh = logging.FileHandler(os.path.join(LOG_DIR, log_filename))
#     fh.setLevel(logging.INFO)
#     fh.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
#     logger.addHandler(fh)

#     return logger
