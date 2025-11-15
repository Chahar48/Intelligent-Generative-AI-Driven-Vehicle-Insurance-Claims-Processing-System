# claim_pipeline/preprocessing/pdf_to_images.py
"""
Beginner-friendly PDF → images converter.
Only uses PyMuPDF to render each page as an image.
No deskew, no OCR enhancements, no debug folders, no hashing.
"""

import os
from pathlib import Path
import fitz  # PyMuPDF
import cv2


def pdf_to_images(pdf_path: str, claim_id: str, output_format="png", dpi=200):
    """
    Converts each PDF page into an image.

    Always returns a list of dictionaries:
    {
        "page_no": int,
        "image_path": "path/to/image",
        "width": int,
        "height": int,
        "success": True/False,
        "notes": "..."
    }
    """

    results = []

    # Create output folder: data/images/<claim_id>
    output_dir = f"data/images/{claim_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Try to open the PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        results.append({
            "page_no": 0,
            "image_path": "",
            "width": 0,
            "height": 0,
            "success": False,
            "notes": "Failed to open PDF"
        })
        return results

    pdf_name = Path(pdf_path).stem

    # Convert each page
    for i in range(len(doc)):
        page_no = i + 1
        try:
            page = doc.load_page(i)

            # Render with DPI scaling
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix)

            image_path = os.path.join(
                output_dir,
                f"{pdf_name}_page_{page_no}.{output_format}"
            )
            pix.save(image_path)

            # Get width & height
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
            else:
                h, w = 0, 0

            results.append({
                "page_no": page_no,
                "image_path": image_path,
                "width": w,
                "height": h,
                "success": True,
                "notes": "Page rendered successfully"
            })

        except Exception:
            results.append({
                "page_no": page_no,
                "image_path": "",
                "width": 0,
                "height": 0,
                "success": False,
                "notes": "Error rendering page"
            })

    return results

