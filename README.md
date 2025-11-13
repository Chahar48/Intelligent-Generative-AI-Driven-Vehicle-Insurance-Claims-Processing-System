# Generative-AI-based-Intelligent-Vehicle-Claims-Processing-System
Vehicle Claims Processing System

📘 Intelligent AI-Driven Insurance Claims Processing System

OCR → Extraction → Normalization → AI Inference → Validation → Decisioning → HITL → Storage → UI

This project implements a production-grade, end-to-end claims processing pipeline that reads insurance documents (PDF, scanned images, emails), extracts structured fields using a combination of OCR + regex + AI, validates them, performs automated triage, executes optional AI reasoning, stores everything for audit, and supports Human-In-The-Loop (HITL) workflows via a Streamlit UI.

🚀 Features
🧠 AI-Powered Extraction

Uses Groq LLM (Llama 3.1 models) for:

Field inference

Claim summarization

AI-assisted decision explanations

📄 Multi-Modal Ingestion

Supports:

PDFs (searchable + scanned)

Images (JPG, PNG)

TXT files

Emails (EML)

🔍 Intelligent Field Extraction

Regex + heuristic extraction

AI inference when fields are missing

Confidence scoring

Normalization of amount, date, email, phone, strings

🛡 Validation Layer

Data consistency checks

Required fields analysis

Estimated vs insured amount logic

Rules to flag suspicious claims

🤖 Decision Engine

Rule-based decisioning

AI reasoning when confidence is low

Produces: approve / reject / review

🧑‍🏫 Human-In-The-Loop (HITL)

Any claim requiring review enters HITL queue

Human can correct fields

Override decision

Add comments

All actions are logged with timestamps

📦 JSON-Based Storage

Stores:

raw file + SHA256 hash

extracted text

structured fields (raw + normalized + AI inference)

validation

summary

final decision

audit logs

HITL tasks

🔐 Security & Privacy

PII redaction in logs

Secure JSON logging

Audit trails

Role-based access (admin, reviewer, auditor, user)

Retention manager to archive/delete old claims

🎨 Streamlit UI

Three pages:

Upload Claim

HITL Review Tasks

Admin (audit viewer, retention tools)

🏗 System Architecture

        ┌────────────────────┐
        │   Streamlit UI     │
        │ upload / review    │
        └─────────┬──────────┘
                  │
         pipeline_runner.py
                  │
 ┌───────────────┼───────────────────────────┐
 │               │                           │
 ▼               ▼                           ▼
OCR        Field Extraction       AI Field Inference
(pdf/email)  regex + heuristics     (Groq LLM)
 │               │                           │
 └──────────► Normalizer ◄────────────────────┘
                     │
                     ▼
                 Validator
                     │
                     ▼
            Decision Engine (rule + AI)
                     │
                     ▼
            Human-in-the-Loop Trigger
                     │
                     ▼
               Final Decision
                     │
                     ▼
              Storage Manager
        (raw → text → fields → decision)


📂 Project Structure
claim_pipeline/
│
├── pipeline_runner.py
│
├── ai/
│   ├── ai_field_infer.py
│   └── summarizer.py
│
├── processing/
│   ├── pdf_extractor.py
│   ├── pdf_to_images.py
│   ├── text_reader.py
│   ├── field_extractor.py
│   ├── normalizer.py
│   ├── validator.py
│   └── decision_engine.py
│
├── hitl/hitl_manager.py
│
├── storage/storage_manager.py
│
├── security/
│   ├── pii_redactor.py
│   ├── auth_manager.py
│   ├── retention_manager.py
│   └── secure_logging.py
│
├── config/
│   ├── config.py
│   └── logging.yaml
│
└── ui/app.py        ← Streamlit App


⚙ Installation & Setup
1️⃣ Clone Repository
git clone <repo_url>
cd <project_root>

2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate            # macOS/Linux
.venv\Scripts\activate               # Windows

3️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

4️⃣ Add Environment Variables

Create a file .env in the project root:

GROQ_API_KEY=your_groq_key
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.2


▶ Running the Pipeline
Run Streamlit UI:
streamlit run claim_pipeline/ui/app.py


This launches:

✔ Upload page
✔ HITL review page
✔ Admin page

🧠 Running Pipeline Programmatically
from claim_pipeline.pipeline_runner import ClaimsPipeline

pipeline = ClaimsPipeline()
result = pipeline.run("path/to/file.pdf")

print(result["decision"])
print(result["summary"])

👩‍🏫 Human-In-The Loop Workflow

A claim enters HITL when:

OCR confidence < threshold

Missing critical fields

Validation raises issues

Decision confidence < 0.75

The human reviewer can:

✔ Edit extracted fields
✔ Correct OCR
✔ Override decision
✔ Approve the claim
✔ Add comments

Every action produces:

data/hitl/<claim_id>.json
data/audit_logs/<timestamp>.json

🔐 Security & Privacy
Implemented:

✅ PII Redaction in Logs
Emails, phones, names, policy numbers masked.

✅ SHA256 Verification
Every uploaded file is hashed.

✅ Secure JSON Logging
Structured, redacted logs.

✅ Role-based Access
(admin, reviewer, auditor, basic user)

👉 Defined in: security/auth_manager.py

🗃 Storage Format
data/
└── claims/
    └── <claim_id>/
        ├── raw/
        ├── extracted/
        ├── fields/
        └── decision/


Storage ensures reproducibility + audit.

🧪 Testing

To run basic tests:

pytest -q


(only if test suite is added — optional)

🐳 Docker Support

Build image:

docker build -t claims-app .


Run:

docker run -p 8501:8501 --env-file .env claims-app


Open browser:

➡ https://localhost:8501

🧭 Roadmap (Optional Extensions)

Add FastAPI REST API

Add CI/CD pipeline

Add MongoDB/PostgreSQL backend

Add document viewer (highlight extracted values)

Add model monitoring dashboard

📝 Conclusion

This repository provides a complete, production-ready Intelligent Claims Processing System capable of:

Reading real-world, noisy insurance claim documents

Extracting structured data

Validating and reasoning over claims

Running AI inference for missing fields and summaries

Making automated decisions

Supporting human corrections

Ensuring auditability, privacy, and security

Offering a clean UI for all workflows

If you want, I can also generate:

📌 API version (FastAPI)
📌 Full unit-test suite
📌 Sample documents dataset
📌 Architecture diagram (PNG/SVG)

Just tell me:
👉 "Generate architecture diagram" or
👉 "Generate test suite"