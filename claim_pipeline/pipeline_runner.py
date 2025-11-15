# claim_pipeline/pipeline_runner.py
"""
Simple, robust, logging-free pipeline runner.

This file orchestrates:
- detection -> extraction -> field extraction -> normalization
- validation -> AI inference -> summarization -> decision -> HITL
All calls are defensive and type-stable.
"""

import os
import uuid
from typing import Dict, Any, List

from claim_pipeline.storage.storage_manager import StorageManager
from claim_pipeline.hitl.hitl_manager import HITLManager

from claim_pipeline.utils.text_reader import read_text_file_str
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


class ClaimsPipeline:
    """
    Minimal, robust pipeline runner without logging.
    """

    def __init__(self):
        self.storage = StorageManager()
        self.hitl = HITLManager()

    # ---------------------------
    # Public API
    # ---------------------------
    def run(self, file_path: str, uploader: str = "system") -> Dict[str, Any]:
        claim_id = f"claim_{uuid.uuid4().hex[:8]}"

        # 1) detect file type
        route = detect_file_type(file_path)
        file_type = route.type if hasattr(route, "type") else getattr(route, "recommended_processor", "unknown")

        # 2) save raw upload (StorageManager handles reading local file)
        try:
            raw_storage = self.storage.save_raw_upload(claim_id, file_path, uploader)
        except Exception:
            raw_storage = None

        # 3) extract text
        raw_text = ""
        if file_type == "pdf":
            # Extract text from PDF (may return dict with 'text')
            try:
                pdf_out = extract_text_from_pdf(file_path)
                pdf_text = pdf_out.get("text") if isinstance(pdf_out, dict) else str(pdf_out or "")
                if not pdf_text.strip():
                    # scanned PDF -> render pages and OCR each
                    pages = pdf_to_images(file_path, claim_id)
                    ocr_texts: List[str] = []
                    for p in pages:
                        # PageImageInfo dataclass -> saved_path attribute or dict key
                        img_path = getattr(p, "saved_path", None) or (p.get("saved_path") if isinstance(p, dict) else None)
                        if img_path:
                            try:
                                ocr_res = ocr_image_best(img_path, claim_id=claim_id)
                                txt = ocr_res.get("text") if isinstance(ocr_res, dict) else ""
                                ocr_texts.append(txt or "")
                            except Exception:
                                ocr_texts.append("")
                    raw_text = "\n".join([t for t in ocr_texts if t])
                else:
                    raw_text = pdf_text
            except Exception:
                raw_text = ""

        elif file_type == "image":
            try:
                ocr_res = ocr_image_best(file_path, claim_id=claim_id)
                raw_text = ocr_res.get("text") if isinstance(ocr_res, dict) else ""
            except Exception:
                raw_text = ""

        elif file_type == "text":
            try:
                raw_text = read_text_file_str(file_path)
            except Exception:
                raw_text = ""

        else:
            # unknown file type -> try to read as text
            try:
                raw_text = read_text_file_str(file_path)
            except Exception:
                raw_text = ""

        # store extracted text
        try:
            text_storage = self.storage.save_extracted_text(claim_id, raw_text)
        except Exception:
            text_storage = None

        # 4) field extraction (rule-based extractor)
        extracted = extract_claim_fields(raw_text) or {}
        # extracted expected shape: {"fields": {...}, "raw_lines": [...], "notes": [...]}

        # 4a) build a simple raw_fields mapping (field -> primitive value) for validators/normalizers
        raw_fields_map: Dict[str, Any] = {}
        fields_section = extracted.get("fields") if isinstance(extracted, dict) else None
        if isinstance(fields_section, dict):
            for k, v in fields_section.items():
                # v may be the _wrap dict or already a primitive; prefer v['value']
                if isinstance(v, dict) and "value" in v:
                    raw_fields_map[k] = v.get("value")
                else:
                    raw_fields_map[k] = v
        else:
            raw_fields_map = {}

        # 5) normalization of fields (returns normalized structures)
        norm_fields = self._normalize_all_fields(raw_fields_map)

        # 6) validation - pass simple raw_fields_map (validator expects mapping of raw values)
        try:
            validation = validate_claim_fields(raw_fields_map)
        except Exception:
            validation = {"issues": ["validation_failed"], "missing_required": [], "is_complete": False, "recommendation": "review"}

        # 7) persist structured fields (raw + normalized + validation); AI inference empty for now
        try:
            self.storage.save_structured_fields(
                claim_id,
                {
                    "raw_fields": extracted,
                    "normalized": norm_fields,
                    "validation": validation,
                    "ai_inference": {}
                }
            )
        except Exception:
            pass

        # 8) AI field inference (accepts the full extracted dict as partial_fields)
        try:
            ai_fields = infer_claim_fields(raw_text, extracted)
            # ensure ai_fields is a dict
            if not isinstance(ai_fields, dict):
                ai_fields = {}
        except Exception:
            ai_fields = {}

        # 9) save fields including ai inference
        try:
            self.storage.save_structured_fields(
                claim_id,
                {
                    "raw_fields": extracted,
                    "normalized": norm_fields,
                    "validation": validation,
                    "ai_inference": ai_fields
                }
            )
        except Exception:
            pass

        # 10) summarization (AI) - if summarizer fails, fallback is used inside it
        try:
            summary = generate_claim_summary(raw_text, ai_fields, validation)
        except Exception:
            summary = {"summary": "", "decision": validation.get("recommendation", "review"), "reasoning": "", "confidence": 0.5, "source": "fallback"}

        # optionally persist summary if storage supports it
        if hasattr(self.storage, "save_summary"):
            try:
                self.storage.save_summary(claim_id, summary)
            except Exception:
                pass

        # 11) decision (uses normalized fields + validator + summary + raw text)
        try:
            decision = decide_claim(
                fields=norm_fields,           # normalized fields for stable decision
                validation=validation,        # validator output
                summary=summary,              # AI summary
                full_text=raw_text            # raw extracted text for fallback checks
            )
        except Exception:
            decision = {
                "decision": "review",
                "confidence": 0.4,
                "requires_human": True,
                "reasons": ["decision_failed"],
                "evidence": {"amount": None, "insured_amount": None, "low_confidence_fields": []},
                "flags": ["engine_error"]
            }

        try:
            self.storage.save_decision(claim_id, decision)
        except Exception:
            pass

        # 12) create HITL task if required
        hitl_required = bool(decision.get("requires_human", False))
        if hitl_required and getattr(self, "hitl", None):
            try:
                task = self.hitl.create_review_task(
                    claim_id=claim_id,
                    raw_text=raw_text,
                    extracted_fields=extracted,
                    validation=validation,
                    ai_inference=ai_fields,
                    summary=summary,
                    decision=decision
                )
                # optional storage for hitl task
                try:
                    self.storage.save_hitl_task(claim_id, task)
                except Exception:
                    pass
            except Exception:
                # creating HITL failed; continue
                pass

        # 13) final response
        return {
            "claim_id": claim_id,
            "raw_text": raw_text,
            "fields_raw_extracted": extracted,
            "fields_normalized": norm_fields,
            "fields_ai": ai_fields,
            "validation": validation,
            "summary": summary,
            "decision": decision,
            "hitl_required": hitl_required,
            "storage": {
                "raw": raw_storage,
                "text": text_storage,
            }
        }

    # ---------------------------
    # Helpers
    # ---------------------------
    def _normalize_all_fields(self, fields: dict) -> dict:
        # fields is a simple mapping name -> primitive value (or None)
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


