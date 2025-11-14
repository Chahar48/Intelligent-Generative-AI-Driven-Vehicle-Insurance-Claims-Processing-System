# claim_pipeline/preprocessing/pdf_to_images.py

import os
import fitz  # PyMuPDF
import cv2
import hashlib
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------
# Dataclass for page metadata
# ---------------------------------------------------------
@dataclass
class PageImageInfo:
    page_number: int
    saved_path: str
    dpi: int
    width: int
    height: int
    sha256: str
    confidence: float
    warnings: List[str]
    debug_variants: Dict[str, str]  # raw, deskewed, thresholded


# ---------------------------------------------------------
# Helper: compute hash
# ---------------------------------------------------------
def compute_sha256(path: str) -> str:
    sha = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha.update(block)
        return sha.hexdigest()
    except Exception as e:
        logger.error(f"[pdf_to_images] Error hashing file {path}: {e}")
        return ""


# ---------------------------------------------------------
# Helper: deskew using Hough transform
# ---------------------------------------------------------
def deskew_image(img: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return img


# ---------------------------------------------------------
# OCR quality enhancements — denoise + CLAHE + threshold
# ---------------------------------------------------------
def enhance_image(img: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(contrast, h=10)
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31, 2
        )
        return thresh
    except Exception:
        return img


# ---------------------------------------------------------
# Auto-rotate if landscape
# ---------------------------------------------------------
def correct_orientation(img: np.ndarray) -> np.ndarray:
    try:
        h, w = img.shape[:2]
        if w > h:  # landscape detected
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img
    except Exception:
        return img


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------
def pdf_to_images(
    pdf_path: str,
    claim_id: str,
    output_format: str = "png",
    dpi: int = 300,
) -> List[PageImageInfo]:
    """
    Convert each PDF page to images with:
    - per-claim folder structure
    - 300 DPI rendering
    - deskewing, denoising, CLAHE enhancement
    - landscape orientation correction
    - debug variants saved
    - SHA256 hashing
    - structured metadata returned
    """

    if output_format not in ["png", "jpg", "jpeg"]:
        logger.error(f"[pdf_to_images] Unsupported output format: {output_format}")
        return []

    # Per-claim folder structure
    output_dir = f"data/images/{claim_id}"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    pdf_name = Path(pdf_path).stem

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"[pdf_to_images] Cannot open PDF {pdf_path}: {e}")
        return []

    for page_index in range(len(doc)):
        warnings = {}
        debug_paths = {}

        try:
            page = doc.load_page(page_index)

            # Render page → raw image
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix)
            raw_path = os.path.join(output_dir, f"{pdf_name}_page_{page_index+1}_raw.{output_format}")
            pix.save(raw_path)
            debug_paths["raw"] = raw_path

            # Load into CV2
            img = cv2.imread(raw_path)
            if img is None:
                logger.error(f"[pdf_to_images] Cannot read rendered image: {raw_path}")
                continue

            # Step: auto-rotate if needed
            img = correct_orientation(img)

            # Step: deskew
            deskewed = deskew_image(img)
            deskew_path = os.path.join(output_dir, f"{pdf_name}_page_{page_index+1}_deskewed.{output_format}")
            cv2.imwrite(deskew_path, deskewed)
            debug_paths["deskewed"] = deskew_path

            # Step: enhance for OCR
            enhanced = enhance_image(deskewed)
            final_path = os.path.join(output_dir, f"{pdf_name}_page_{page_index+1}.{output_format}")
            cv2.imwrite(final_path, enhanced)

            debug_paths["thresholded"] = final_path

            # Resolution warnings
            h, w = enhanced.shape[:2]
            low_res_warning = (w < 800 or h < 800)
            warning_msgs = []
            if low_res_warning:
                warning_msgs.append("Low resolution — OCR accuracy may be reduced.")

            # Compute hash
            sha = compute_sha256(final_path)

            # Confidence scoring (simple heuristic)
            conf = 0.8
            if low_res_warning:
                conf -= 0.2

            result = PageImageInfo(
                page_number=page_index + 1,
                saved_path=final_path,
                dpi=dpi,
                width=w,
                height=h,
                sha256=sha,
                confidence=conf,
                warnings=warning_msgs,
                debug_variants=debug_paths
            )

            results.append(result)
            logger.info(f"[pdf_to_images] Page {page_index+1} rendered → {final_path}")

        except Exception as e:
            logger.error(f"[pdf_to_images] Error rendering page {page_index+1}: {e}")
            continue  # Skip corrupt page instead of failing

    return results