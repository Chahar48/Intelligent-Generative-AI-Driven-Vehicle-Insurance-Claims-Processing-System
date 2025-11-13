# claim_pipeline/extraction/ocr_extractor.py
"""
Robust OCR extractor combining PaddleOCR + Tesseract with preprocessing, retries,
debug artifacts, SHA256 hashing, and a structured output schema.

Main entrypoint:
    ocr_image_best(image_path, claim_id=None, use_gpu=False, confidence_threshold=0.6, max_retries=2)

Returns a dict:
{
  "text": "<full_text>",
  "lines": [ {"text": "...", "bbox":[x,y,w,h], "conf":0.92, "engine":"paddle"}, ... ],
  "engine": "paddle"|"tesseract"|"fusion",
  "confidence": 0.87,            # overall confidence
  "per_engine": {
      "paddle": {"text":"...", "confidence": 0.88, "lines":[...], "error": None},
      "tesseract": {...}
  },
  "image_sha256": "...",
  "debug": {
      "paths": {"raw": "...", "rotated": "...", "variant_adaptive": "...", ...},
      "logs": [...]
  },
  "warnings": ["low_confidence", "blur_detected"],
  "metadata": {"width":1024, "height":768}
}
"""

import os
import io
import cv2
import time
import hashlib
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import uuid

# Optional PaddleOCR
try:
    from paddleocr import PaddleOCR
    _paddle_available = True
except Exception:
    PaddleOCR = None
    _paddle_available = False

import pytesseract  # Tesseract wrapper (assumed installed)

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def compute_sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def compute_sha256_file(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return compute_sha256_bytes(f.read())
    except Exception as e:
        logger.warning(f"[ocr_extractor] sha256 failed for {path}: {e}")
        return ""


def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)


def save_debug_image(img: np.ndarray, claim_id: Optional[str], tag: str) -> str:
    """
    Save debug image inside per-claim folder if claim_id provided,
    otherwise save under data/images/debug.
    Returns saved path.
    """
    base_dir = f"data/images/{claim_id}/ocr_debug" if claim_id else "data/images/ocr_debug"
    ensure_folder(base_dir)
    fname = f"{tag}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(base_dir, fname)
    try:
        cv2.imwrite(path, img)
    except Exception as e:
        logger.error(f"[ocr_extractor] Failed to save debug image {path}: {e}")
    return path


# -------------------------------------------------------------------------
# Preprocessing helpers
# -------------------------------------------------------------------------
def load_image_cv2(image_path: str, max_dim: int = 3000) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def detect_blur(image: np.ndarray, threshold: float = 100.0) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < threshold


def deskew_image(img: np.ndarray) -> np.ndarray:
    """
    Basic deskew using rotation from minAreaRect on thresholded image.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.shape[0] < 10:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return img


def dewarp_image(img: np.ndarray) -> np.ndarray:
    """
    Attempt to detect document contour and apply perspective transform.
    Best-effort; if fails, returns original.
    """
    try:
        orig = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                # order points
                rect = order_points(pts)
                (tl, tr, br, bl) = rect
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxWidth = max(int(widthA), int(widthB))
                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxHeight = max(int(heightA), int(heightB))
                dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
                return warped
        return img
    except Exception:
        return img


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def enhance_variants(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Produce multiple variants for OCR: adaptive, clahe, bilateral, upscale, gamma.
    Returns dict[tag] = image(np.ndarray)
    """
    variants = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # adaptive threshold on resized image
    try:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        variants["adaptive"] = th
    except Exception:
        pass

    # CLAHE + Otsu
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        _, th2 = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["clahe"] = th2
    except Exception:
        pass

    # Bilateral + Otsu
    try:
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["bilateral"] = th3
    except Exception:
        pass

    # Upscale + adaptive
    try:
        h, w = gray.shape
        up = cv2.resize(gray, (min(3000, int(w * 1.5)), min(3000, int(h * 1.5))), interpolation=cv2.INTER_CUBIC)
        up_th = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 2)
        variants["upscale_adaptive"] = up_th
    except Exception:
        pass

    # Gamma correction variant
    try:
        gamma = 1.5
        look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_img = cv2.LUT(gray, look_up_table)
        _, gamma_th = cv2.threshold(gamma_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["gamma"] = gamma_th
    except Exception:
        pass

    return variants


# -------------------------------------------------------------------------
# OCR Engines wrappers
# -------------------------------------------------------------------------
def _paddle_ocr_engine(var_img_path: str, paddle_ocr: Any) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    Run PaddleOCR on an image file and return (text, mean_conf, lines_list).
    lines_list contains dicts: {'text':..., 'bbox':[x_min,y_min,x_max,y_max], 'conf':...}
    """
    try:
        raw_res = paddle_ocr.ocr(var_img_path, cls=True)
        lines = []
        confidences = []
        for block in raw_res:
            # raw_res structure: [[(box, (text, conf)), ...], ...] for some versions
            # For safety, iterate and extract tuples
            if isinstance(block, list):
                for item in block:
                    if len(item) >= 2:
                        bbox = item[0]  # 4 points
                        text_conf = item[1]
                        txt = text_conf[0]
                        conf = float(text_conf[1]) if text_conf[1] is not None else 0.0
                        # convert bbox 4 points to [x_min, y_min, x_max, y_max]
                        xs = [int(p[0]) for p in bbox]
                        ys = [int(p[1]) for p in bbox]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        lines.append({"text": txt, "bbox": [x_min, y_min, x_max - x_min, y_max - y_min], "conf": conf})
                        confidences.append(conf)
            else:
                # fallback for other formats
                pass
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        full_text = "\n".join([l["text"] for l in lines]).strip()
        return full_text, mean_conf, lines
    except Exception as e:
        logger.error(f"[ocr_extractor] PaddleOCR engine error: {e}")
        return "", 0.0, []


def _tesseract_engine(var_img: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    Use pytesseract.image_to_data to get words/lines + confidences + boxes.
    Input can be numpy image (grayscale or color).
    """
    try:
        pil = Image.fromarray(var_img)
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)

        n = len(data["level"])
        lines = []
        confidences = []
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = 0.0
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            lines.append({"text": text, "bbox": [x, y, w, h], "conf": conf})
            confidences.append(conf)
        full_text = " ".join([l["text"] for l in lines]).strip()
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return full_text, mean_conf, lines
    except Exception as e:
        logger.error(f"[ocr_extractor] Tesseract engine error: {e}")
        return "", 0.0, []


# -------------------------------------------------------------------------
# Fusion and selection logic
# -------------------------------------------------------------------------
def _select_best_engine(paddle_out: Dict[str, Any], tesseract_out: Dict[str, Any]) -> Tuple[str, float, str]:
    """
    Simple selection/fusion logic:
    - Prefer engine with higher mean confidence
    - If both low, try simple fusion (concatenate unique lines)
    Returns (chosen_engine, chosen_confidence, reason)
    """
    p_conf = paddle_out.get("confidence", 0.0)
    t_conf = tesseract_out.get("confidence", 0.0)

    if p_conf >= t_conf and p_conf > 0:
        return "paddle", p_conf, "paddle_higher_conf"
    if t_conf > p_conf and t_conf > 0:
        return "tesseract", t_conf, "tesseract_higher_conf"

    # if both zero or very low, fallback to whichever produced more text
    p_len = len(paddle_out.get("text", "") or "")
    t_len = len(tesseract_out.get("text", "") or "")
    if p_len >= t_len and p_len > 0:
        return "paddle", p_conf, "paddle_longer_text"
    if t_len > p_len and t_len > 0:
        return "tesseract", t_conf, "tesseract_longer_text"

    return "none", 0.0, "no_output"


# -------------------------------------------------------------------------
# Handwriting detection (heuristic)
# -------------------------------------------------------------------------
def detect_handwriting(lines: List[Dict[str, Any]]) -> bool:
    """
    Heuristic: if average confidence is low and average height of bounding boxes is small/variable,
    this can indicate handwriting. This is a heuristic; use as a signal only.
    """
    if not lines:
        return False
    confs = [l.get("conf", 0.0) for l in lines if l.get("conf") is not None]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    heights = [l["bbox"][3] for l in lines if l.get("bbox")]
    mean_h = float(np.mean(heights)) if heights else 0.0
    return mean_conf < 40.0 and mean_h < 30  # thresholds tuned heuristically


# -------------------------------------------------------------------------
# Main high-level OCR function
# -------------------------------------------------------------------------
def ocr_image_best(
    image_path: str,
    claim_id: Optional[str] = None,
    use_gpu: bool = False,
    confidence_threshold: float = 0.6,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    High-level function to run OCR on an image and return a structured result.
    - image_path: local path to image
    - claim_id: optional claim id to place debug files
    - use_gpu: if True, will try to initialize PaddleOCR with GPU (if available)
    - confidence_threshold: below this warn/flag for human-in-loop
    - max_retries: number of retries for PaddleOCR failures
    """

    start_time = time.time()
    debug = {"paths": {}, "logs": []}
    warnings = []
    metadata = {}

    # Load image and compute hash
    try:
        img = load_image_cv2(image_path)
        h, w = img.shape[:2]
        metadata["width"] = int(w)
        metadata["height"] = int(h)
    except Exception as e:
        logger.error(f"[ocr_extractor] Cannot load image {image_path}: {e}")
        return {"text": "", "lines": [], "engine": None, "confidence": 0.0, "per_engine": {}, "image_sha256": "", "debug": debug, "warnings": ["load_error"], "metadata": {}}

    image_sha = compute_sha256_file(image_path)
    debug["paths"]["raw"] = image_path
    metadata["image_sha256"] = image_sha

    # Basic image quality checks
    if detect_blur(img):
        warnings.append("blur_detected")
    if max(img.shape[:2]) < 800:
        warnings.append("low_resolution")

    # Preprocessing: deskew, dewarp, rotation correction
    rotated = deskew_image(img)
    if rotated is not None and not np.array_equal(rotated, img):
        p = save_debug_image(rotated, claim_id, "deskewed")
        debug["paths"]["deskewed"] = p
        debug["logs"].append("deskew_applied")
        img_proc = rotated
    else:
        img_proc = img

    dewarped = dewarp_image(img_proc)
    if dewarped is not None and not np.array_equal(dewarped, img_proc):
        p = save_debug_image(dewarped, claim_id, "dewarped")
        debug["paths"]["dewarped"] = p
        debug["logs"].append("dewarp_applied")
        img_proc = dewarped

    # Generate enhancement variants
    variants = enhance_variants(img_proc)
    # Save variant images and track paths
    variant_paths = {}
    variant_imgs = {}
    for tag, vimg in variants.items():
        # ensure vimg is single-channel (cv2) — convert to 3-channel for tesseract if needed
        if len(vimg.shape) == 2:
            v_to_save = cv2.cvtColor(vimg, cv2.COLOR_GRAY2BGR)
        else:
            v_to_save = vimg
        pth = save_debug_image(v_to_save, claim_id, f"variant_{tag}")
        variant_paths[tag] = pth
        variant_imgs[tag] = vimg

    debug["paths"].update(variant_paths)

    # Initialize PaddleOCR if available with desired GPU flag
    paddle_ocr_instance = None
    if _paddle_available:
        try:
            paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
        except Exception as e:
            logger.warning(f"[ocr_extractor] PaddleOCR init failed (gpu={use_gpu}): {e}")
            paddle_ocr_instance = None

    # Run OCR on each variant, collect per-engine outputs
    per_engine = {"paddle": {"text": "", "confidence": 0.0, "lines": [], "error": None},
                  "tesseract": {"text": "", "confidence": 0.0, "lines": [], "error": None}}

    # PaddleOCR (retry logic)
    if paddle_ocr_instance is not None:
        paddle_attempts = 0
        while paddle_attempts <= max_retries:
            try:
                # run on best-looking variant first (adaptive if exists)
                ordered_tags = ["adaptive", "clahe", "upscale_adaptive", "bilateral", "gamma"]
                for tag in ordered_tags:
                    if tag not in variant_paths:
                        continue
                    path = variant_paths[tag]
                    txt, conf, lines = _paddle_ocr_engine(path, paddle_ocr_instance)
                    if txt:
                        per_engine["paddle"]["text"] = txt
                        per_engine["paddle"]["confidence"] = conf
                        per_engine["paddle"]["lines"] = lines
                        break
                # if nothing matched, try any variant
                if not per_engine["paddle"]["text"]:
                    for tag, path in variant_paths.items():
                        txt, conf, lines = _paddle_ocr_engine(path, paddle_ocr_instance)
                        if txt:
                            per_engine["paddle"]["text"] = txt
                            per_engine["paddle"]["confidence"] = conf
                            per_engine["paddle"]["lines"] = lines
                            break
                break  # success or empty but no exception
            except Exception as e:
                paddle_attempts += 1
                per_engine["paddle"]["error"] = str(e)
                logger.warning(f"[ocr_extractor] PaddleOCR attempt {paddle_attempts} failed: {e}")
                time.sleep(0.5)
        if per_engine["paddle"]["confidence"] == 0.0:
            logger.info("[ocr_extractor] Paddle produced no confident output.")
    else:
        per_engine["paddle"]["error"] = "PaddleOCR not available"

    # Tesseract on best variant(s)
    try:
        # prefer adaptive/clahe then others
        for tag in ["adaptive", "clahe", "upscale_adaptive", "bilateral", "gamma"]:
            if tag not in variant_imgs:
                continue
            # convert to 3-channel if needed for tesseract wrapper
            vimg = variant_imgs[tag]
            if len(vimg.shape) == 2:
                v_for_tesseract = cv2.cvtColor(vimg, cv2.COLOR_GRAY2BGR)
            else:
                v_for_tesseract = vimg
            txt, conf, lines = _tesseract_engine(v_for_tesseract)
            if txt:
                per_engine["tesseract"]["text"] = txt
                per_engine["tesseract"]["confidence"] = conf
                per_engine["tesseract"]["lines"] = lines
                break
        if not per_engine["tesseract"]["text"]:
            # try any remaining variant
            for tag, vimg in variant_imgs.items():
                if len(vimg.shape) == 2:
                    v_for_tesseract = cv2.cvtColor(vimg, cv2.COLOR_GRAY2BGR)
                else:
                    v_for_tesseract = vimg
                txt, conf, lines = _tesseract_engine(v_for_tesseract)
                if txt:
                    per_engine["tesseract"]["text"] = txt
                    per_engine["tesseract"]["confidence"] = conf
                    per_engine["tesseract"]["lines"] = lines
                    break
    except Exception as e:
        per_engine["tesseract"]["error"] = str(e)
        logger.warning(f"[ocr_extractor] Tesseract failed: {e}")

    # Select best engine or attempt basic fusion
    chosen_engine, chosen_conf, reason = _select_best_engine(per_engine["paddle"], per_engine["tesseract"])
    debug["logs"].append({"selection_reason": reason})

    # Build unified lines list: prefer chosen engine lines, fallback to other
    unified_lines = []
    if chosen_engine == "paddle":
        unified_lines = per_engine["paddle"].get("lines", [])
        overall_text = per_engine["paddle"].get("text", "")
    elif chosen_engine == "tesseract":
        unified_lines = per_engine["tesseract"].get("lines", [])
        overall_text = per_engine["tesseract"].get("text", "")
    else:
        # fusion: combine unique lines by text
        seen = set()
        for eng in ("paddle", "tesseract"):
            for l in per_engine[eng].get("lines", []):
                t = l.get("text", "").strip()
                if t and t not in seen:
                    seen.add(t)
                    l_copy = l.copy()
                    l_copy["engine"] = eng
                    unified_lines.append(l_copy)
        overall_text = "\n".join([l["text"] for l in unified_lines])

    # Tag unified lines with engine if missing
    for l in unified_lines:
        if "engine" not in l:
            l["engine"] = chosen_engine

    # Heuristic: if handwriting detected, add warning
    if detect_handwriting(unified_lines):
        warnings.append("handwriting_detected")

    # Final confidence is chosen_conf but penalize for warnings
    final_conf = float(chosen_conf or 0.0)
    if "blur_detected" in warnings:
        final_conf = max(0.0, final_conf - 0.15)
    if "low_resolution" in warnings:
        final_conf = max(0.0, final_conf - 0.15)
    if "handwriting_detected" in warnings:
        final_conf = max(0.0, final_conf - 0.2)

    # Human-in-loop suggested
    needs_human = final_conf < confidence_threshold

    end_time = time.time()
    runtime = end_time - start_time

    # Prepare result schema
    result = {
        "text": overall_text,
        "lines": unified_lines,
        "engine": chosen_engine,
        "confidence": round(final_conf, 3),
        "per_engine": per_engine,
        "image_sha256": image_sha,
        "debug": debug,
        "warnings": warnings,
        "metadata": {
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "runtime_sec": round(runtime, 2),
            "needs_human": needs_human
        }
    }

    logger.info(f"[ocr_extractor] OCR finished for {image_path} - engine={chosen_engine} conf={result['confidence']} warnings={warnings}")
    return result


# -------------------------------------------------------------------------
# Backward-compatible wrapper
# -------------------------------------------------------------------------
def ocr_image(image_path: str, claim_id: Optional[str] = None, prefer: str = "paddle", **kwargs):
    """
    Back-compatible wrapper for previous code. Calls ocr_image_best.
    """
    res = ocr_image_best(image_path, claim_id=claim_id, use_gpu=kwargs.get("use_gpu", False),
                         confidence_threshold=kwargs.get("confidence_threshold", 0.6),
                         max_retries=kwargs.get("max_retries", 2))
    logger.info(f"[OCR Wrapper] Used engine={res.get('engine')} conf={res.get('confidence')}")
    return res
