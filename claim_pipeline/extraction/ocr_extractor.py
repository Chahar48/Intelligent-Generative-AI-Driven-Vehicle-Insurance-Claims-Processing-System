# claim_pipeline/extraction/ocr_extractor.py
"""
Improved OCR extractor (Tesseract-only)
---------------------------------------

GOALS:
- Preserve SAME output structure so pipeline never breaks
- Improve OCR accuracy significantly using:
    ✓ auto-rotation
    ✓ deskew
    ✓ noise removal
    ✓ bilateral filtering
    ✓ CLAHE contrast enhancement
    ✓ sharpening filters
    ✓ adaptive threshold variants
    ✓ multi-variant OCR and choose best
    ✓ confidence scoring

OUTPUT FORMAT (unchanged):
{
    "text": str,
    "lines": [ {"text": str, "bbox":[x,y,w,h], "conf": float}, ... ],
    "engine": "tesseract",
    "confidence": float
}
"""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image
import pytesseract

# -------------------------------------------------------------------
# Safe helpers
# -------------------------------------------------------------------

def _empty():
    return {"text": "", "lines": [], "engine": "tesseract", "confidence": 0.0}


def _read_image(path: str):
    try:
        return cv2.imread(path)
    except:
        return None


# -------------------------------------------------------------------
# Image enhancement methods
# -------------------------------------------------------------------

def _deskew(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except:
        return img


def _denoise(img):
    try:
        return cv2.bilateralFilter(img, 9, 75, 75)
    except:
        return img


def _clahe(img):
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except:
        return img


def _sharpen(img):
    try:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    except:
        return img


def _adaptive(img):
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(g, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   31, 2)
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    except:
        return img


def _rotate_90(img):
    try:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    except:
        return img


# -------------------------------------------------------------------
# Tesseract OCR wrapper
# -------------------------------------------------------------------

def _run_tesseract(img) -> Dict[str, Any]:
    try:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        data = pytesseract.image_to_data(
            pil,
            output_type=pytesseract.Output.DICT
        )
    except:
        return _empty()

    lines = []
    confs = []

    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except:
            conf = 0.0

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        lines.append({
            "text": text,
            "bbox": [x, y, w, h],
            "conf": conf
        })
        confs.append(conf)

    full_text = " ".join([l["text"] for l in lines]).strip()
    mean_conf = float(np.mean(confs)) if confs else 0.0

    return {
        "text": full_text,
        "lines": lines,
        "engine": "tesseract",
        "confidence": round(mean_conf / 100.0, 3)
    }


# -------------------------------------------------------------------
# MAIN OCR PIPELINE (IMPROVED)
# -------------------------------------------------------------------

def ocr_image_best(image_path: str, claim_id: Optional[str] = None) -> Dict[str, Any]:
    img = _read_image(image_path)
    if img is None:
        return _empty()

    # Step 1 — deskew
    v1 = _deskew(img)

    # Step 2 — denoise
    v2 = _denoise(v1)

    # Step 3 — contrast boost
    v3 = _clahe(v2)

    # Step 4 — sharpen
    v4 = _sharpen(v3)

    # Step 5 — adaptive threshold variant
    v5 = _adaptive(v4)

    # Step 6 — rotated variant for upside-down text
    v6 = _rotate_90(v4)

    variants = [v1, v2, v3, v4, v5, v6]

    results = []
    for var in variants:
        results.append(_run_tesseract(var))

    # choose result with highest confidence
    best = max(results, key=lambda x: x.get("confidence", 0))

    return best


# backward compatibility
def ocr_image(image_path: str, claim_id: Optional[str] = None):
    return ocr_image_best(image_path, claim_id=claim_id)


###################################Original code but not that much effective###########################################################

# claim_pipeline/extraction/ocr_extractor.py
"""
Defensive, beginner-friendly OCR module.

Exports:
- ocr_image_best(image_path: str, claim_id: str | None = None) -> dict

Behavior:
- Uses Tesseract if available.
- If Tesseract (or required libs) are missing or fail, returns a safe empty result.
- Never raises at import time.
- Always returns a dictionary with stable types (strings, lists, floats).
"""

from typing import Dict, Any, List, Optional

# Try optional imports very defensively
_cv2_available = False
_pil_available = False
_tesseract_available = False
_np_available = False

try:
    import cv2
    _cv2_available = True
except Exception:
    _cv2_available = False

try:
    from PIL import Image
    _pil_available = True
except Exception:
    _pil_available = False

try:
    import pytesseract
    _tesseract_available = True
except Exception:
    _tesseract_available = False

try:
    import numpy as np
    _np_available = True
except Exception:
    _np_available = False


# -------------------------
# Internal helpers
# -------------------------
def _empty_result() -> Dict[str, Any]:
    return {"text": "", "lines": [], "engine": "none", "confidence": 0.0}


def _read_image_as_array(path: str):
    """
    Read image into a numpy array (BGR as cv2 does).
    Returns None on failure.
    """
    if _cv2_available:
        try:
            img = cv2.imread(path)
            return img
        except Exception:
            return None

    # fallback: use PIL to open and convert to RGB array
    if _pil_available and _np_available:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                arr = np.array(im)
                # convert RGB -> BGR if downstream expects cv2 style (not strictly necessary here)
                if arr.shape[-1] == 3:
                    arr = arr[:, :, ::-1]
                return arr
        except Exception:
            return None

    return None


def _run_tesseract_on_array(arr) -> Dict[str, Any]:
    """
    Run pytesseract on a numpy array image.
    Returns dict with text, lines and confidence.
    """
    if not _tesseract_available or arr is None:
        return _empty_result()

    # Convert array to PIL image if needed
    try:
        if _pil_available:
            # if arr is BGR (cv2), convert to RGB for PIL
            try:
                import numpy as _np  # small local import to avoid name errors if np missing
                if isinstance(arr, _np.ndarray) and arr.shape[-1] == 3:
                    rgb = arr[:, :, ::-1]
                else:
                    rgb = arr
                pil = Image.fromarray(rgb)
            except Exception:
                # last resort: try direct open via PIL - but we only have array here
                pil = None
        else:
            pil = None
    except Exception:
        pil = None

    try:
        if pil is None and _pil_available:
            # try convert via numpy -> PIL again (defensive)
            try:
                pil = Image.fromarray(arr)
            except Exception:
                pil = None
    except Exception:
        pil = None

    # If PIL image isn't available but pytesseract still accepts file path, we will convert below
    try:
        # Use pytesseract.image_to_data to get words + conf + bbox
        if pil is not None:
            data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
        else:
            # If pil not available, try converting arr to string via cv2 imencode and then to PIL
            if _cv2_available:
                try:
                    success, enc = cv2.imencode('.png', arr)
                    if success:
                        from io import BytesIO
                        pil = Image.open(BytesIO(enc.tobytes()))
                        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
                    else:
                        return _empty_result()
                except Exception:
                    return _empty_result()
            else:
                return _empty_result()

        lines = []
        confs: List[float] = []

        n = len(data.get("text", []))
        for i in range(n):
            txt = (data.get("text", [])[i] or "").strip()
            if not txt:
                continue
            # confidence may be non-numeric occasionally
            try:
                conf_raw = data.get("conf", [])[i]
                conf = float(conf_raw) if conf_raw not in (None, "") else 0.0
            except Exception:
                conf = 0.0

            left = int(data.get("left", [])[i] or 0)
            top = int(data.get("top", [])[i] or 0)
            width = int(data.get("width", [])[i] or 0)
            height = int(data.get("height", [])[i] or 0)

            lines.append({
                "text": txt,
                "bbox": [left, top, width, height],
                "conf": float(conf)
            })
            confs.append(conf)

        full_text = " ".join([l["text"] for l in lines]).strip()
        mean_conf = float(sum(confs) / len(confs)) if confs else 0.0

        # map mean_conf (which is typically 0-100) to 0.0-1.0 if it looks like percent
        if mean_conf > 1.0:
            mean_conf_norm = min(1.0, mean_conf / 100.0)
        else:
            mean_conf_norm = mean_conf

        return {
            "text": full_text,
            "lines": lines,
            "engine": "tesseract",
            "confidence": round(float(mean_conf_norm), 3)
        }
    except Exception:
        return _empty_result()


# -------------------------
# Public API
# -------------------------
def ocr_image_best(image_path: str, claim_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Beginner-friendly OCR entrypoint used by pipeline_runner.
    - image_path: path to image file
    - claim_id: optional, ignored here but kept for compatibility
    Returns a stable dict (never raises).
    """
    try:
        arr = _read_image_as_array(image_path)
        if arr is None:
            return _empty_result()

        # Use Tesseract only (as requested). If it's not available, return safe empty.
        if _tesseract_available:
            return _run_tesseract_on_array(arr)

        # If tesseract not available, return deterministic empty result
        return _empty_result()
    except Exception:
        # Last-resort safety: never let exceptions propagate
        return _empty_result()

