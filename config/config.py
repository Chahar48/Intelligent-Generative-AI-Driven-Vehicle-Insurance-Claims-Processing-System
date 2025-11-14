"""
Global configuration for the Claims Processing Pipeline.
All modules should import from here — no hardcoded paths anywhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Base Directories
# ------------------------------
BASE_DIR = "data/"
CLAIMS_DIR = os.path.join(BASE_DIR, "claims")
TMP_DIR = os.path.join(BASE_DIR, "tmp")
LOG_DIR = "logs/"

# Ensure directories exist
os.makedirs(CLAIMS_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------------
# File Storage Subdirectories
# ------------------------------
RAW_DIR = "raw"
EXTRACTED_DIR = "extracted"
STRUCTURED_DIR = "fields"
DECISION_DIR = "decision"
HITL_DIR = "hitl"
AUDIT_DIR = "audit"

# Actual directory where raw uploaded claim files will be stored
RAW_DATA_DIR = CLAIMS_DIR

# ------------------------------
# OCR and Processing Config
# ------------------------------
OCR_DPI = 300
OCR_CONFIDENCE_THRESHOLD = 0.60
FIELD_CONFIDENCE_THRESHOLD = 0.50
DECISION_CONFIDENCE_THRESHOLD = 0.75

# ------------------------------
# LLM Configuration (Groq)
# ------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------------------
# Logging
# ------------------------------
LOGGING_CONFIG = "config/logging.yaml"

# ------------------------------
# Feature Toggles
# ------------------------------
ENABLE_AI_INFERENCE = True
ENABLE_AI_SUMMARIZER = True
ENABLE_AI_REASONING = True
ENABLE_HITL = True     # Turn off to auto-approve low confidence items

# ------------------------------
# Debug Mode
# ------------------------------
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
