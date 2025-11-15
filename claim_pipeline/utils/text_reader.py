# claim_pipeline/utils/text_reader.py
"""
Simple text file reader.
Beginner-friendly, consistent output, no logging, no hashing.
"""

import os


def read_text_file(file_path: str):
    """
    Reads any .txt file and returns:

    {
        "success": True/False,
        "text": "full text",
        "lines": ["...", "..."],
        "encoding": "utf-8" | "latin-1" | etc,
        "size": int,
        "notes": "..."
    }
    """

    result = {
        "success": False,
        "text": "",
        "lines": [],
        "encoding": "",
        "size": 0,
        "notes": ""
    }

    # File exists?
    if not os.path.isfile(file_path):
        result["notes"] = "File not found"
        return result

    # File size
    try:
        size = os.path.getsize(file_path)
        result["size"] = size
    except:
        result["notes"] = "Unable to read file size"
        return result

    # Try simple UTF-8 reading first
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        result["encoding"] = "utf-8"
    except:
        # fallback to latin-1 (reads anything)
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
            result["encoding"] = "latin-1"
        except:
            result["notes"] = "Unable to read file"
            return result

    # Clean and split
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    result["success"] = True
    result["text"] = text
    result["lines"] = [ln.strip() for ln in text.split("\n") if ln.strip()]
    result["notes"] = "File read successfully"

    return result


# def read_text_file_str(file_path: str) -> str:
#     """
#     Returns only the text from the file.
#     """
#     res = read_text_file(file_path)
#     return res["text"]
def read_text_file_str(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = text.replace("\ufeff", "")  # remove BOM  
    return text
