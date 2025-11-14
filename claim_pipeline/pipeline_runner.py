# claim_pipeline/pipeline_runner.py

import os
import uuid
import logging

from claim_pipeline.security.secure_logging import get_logger
from claim_pipeline.storage.storage_manager import StorageManager
from claim_pipeline.hitl.hitl_manager import HITLManager

from claim_pipeline.utils.text_reader import read_text_file
from claim_pipeline.preprocessing.pdf_to_images import pdf_to_images
from claim_pipeline.extraction.pdf_extractor import extract_text_from_pdf
from claim_pipeline.extraction.ocr_extractor import ocr_image_best
from claim_pipeline.processing.field_extractor import extract_claim_fields
from claim_pipeline.processing.normalizer import (
    normalize_amount, normalize_date, normalize_phone,
    normalize_email, normalize_string
)
from claim_pipeline.processing.validator import validate_claim_fields
from claim_pipeline.processing.ai_field_infer import infer_claim_fields
from claim_pipeline.ai.summarizer import generate_claim_summary
from claim_pipeline.processing.decision_engine import decide_claim

from claim_pipeline.ingestion.file_router import detect_file_type
from claim_pipeline.ingestion.uploader import save_uploaded_files


class ClaimsPipeline:
    """
    Master orchestrator of the AI Claims Pipeline.
    Runs all subsystems in correct order + HITL + storage.
    """

    def __init__(self):
        self.logger = get_logger("pipeline_runner")
        self.storage = StorageManager()
        self.hitl = HITLManager()

    # ---------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------
    def run(self, file_path: str, uploader: str = "system") -> dict:
        """
        Run the entire pipeline on one uploaded file.
        Returns full structured result for UI or API.
        """

        claim_id = f"claim_{uuid.uuid4().hex[:8]}"
        self.logger.info(f"[Pipeline] Starting processing for {claim_id}")

        # -----------------------------------------------------
        # 1. Detect File Type
        # -----------------------------------------------------
        info = detect_file_type(file_path)
        file_type = info["type"]
        self.logger.info(f"[Pipeline] Detected file type: {file_type}")

        # Store raw upload
        raw_storage = self.storage.save_raw_upload(claim_id, file_path, uploader)

        # -----------------------------------------------------
        # 2. Extract Text (PDF, Image, or TXT)
        # -----------------------------------------------------
        raw_text = ""

        try:
            if file_type == "pdf":
                pdf_text_data = extract_text_from_pdf(file_path)

                if not pdf_text_data["text"].strip():
                    # PDF scanned → OCR fallback
                    images = pdf_to_images(file_path, output_dir=f"data/tmp/{claim_id}")
                    ocr_text_list = []

                    for img in images:
                        ocr_res = ocr_image_best(img)
                        ocr_text_list.append(ocr_res["text"])

                    raw_text = "\n".join(ocr_text_list)
                else:
                    raw_text = pdf_text_data["text"]

            elif file_type == "image":
                ocr_res = ocr_image_best(file_path)
                raw_text = ocr_res["text"]

            elif file_type == "text":
                raw_text = read_text_file(file_path)

            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            self.logger.error(f"[Pipeline] Extraction error: {e}", exc_info=True)
            raw_text = ""

        # Store extracted text
        text_storage = self.storage.save_extracted_text(claim_id, raw_text)

        # -----------------------------------------------------
        # 3. Field Extraction (Rule-based)
        # -----------------------------------------------------
        extracted_fields = extract_claim_fields(raw_text)
        norm_fields = self._normalize_all_fields(extracted_fields)
        validation = validate_claim_fields(extracted_fields)

        self.storage.save_structured_fields(claim_id, {
            "rule_based": extracted_fields,
            "normalized": norm_fields,
            "validation": validation
        })

        # -----------------------------------------------------
        # 4. AI Field Inference (LLM)
        # -----------------------------------------------------
        ai_fields = infer_claim_fields(raw_text, extracted_fields)
        self.storage.save_structured_fields(claim_id, {"ai_inference": ai_fields})

        # -----------------------------------------------------
        # 5. Summarizer (AI)
        # -----------------------------------------------------
        summary = generate_claim_summary(raw_text, ai_fields, validation)
        self.storage.save_summary(claim_id, summary)

        # -----------------------------------------------------
        # 6. Decision Engine
        # -----------------------------------------------------
        decision = decide_claim(ai_fields, validation, summary)
        self.storage.save_decision(claim_id, decision)

        # -----------------------------------------------------
        # 7. Check if Human Review Required
        # -----------------------------------------------------
        hitl_required = decision.get("requires_human", False)

        if hitl_required:
            task = self.hitl.create_hitl_task(
                claim_id=claim_id,
                stage="final_decision_review",
                payload={
                    "fields": ai_fields,
                    "validation": validation,
                    "summary": summary,
                    "decision": decision,
                }
            )
            self.storage.save_hitl_task(claim_id, task)

        # -----------------------------------------------------
        # 8. Assemble Response
        # -----------------------------------------------------
        return {
            "claim_id": claim_id,
            "raw_text": raw_text,
            "fields": ai_fields,
            "validation": validation,
            "summary": summary,
            "decision": decision,
            "hitl_required": hitl_required,
            "storage": {
                "raw": raw_storage,
                "text": text_storage,
            }
        }

    # ---------------------------------------------------------
    # Helper — Normalize ALL fields
    # ---------------------------------------------------------
    def _normalize_all_fields(self, fields: dict) -> dict:
        """
        Normalize each known field using normalizer_utils
        """
        return {
            "claim_id": normalize_string(fields.get("claim_id")),
            "policy_no": normalize_string(fields.get("policy_no")),
            "claimant_name": normalize_string(fields.get("claimant_name")),
            "phone": normalize_phone(fields.get("phone")),
            "email": normalize_email(fields.get("email")),
            "vehicle": normalize_string(fields.get("vehicle")),
            "damage": normalize_string(fields.get("damage")),
            "date": normalize_date(fields.get("date")),
            "amount_estimated": normalize_amount(fields.get("amount_estimated")),
            "insured_amount": normalize_amount(fields.get("insured_amount")),
        }
