# claim_pipeline/ingestion/uploader.py

import os
import uuid
from pathlib import Path
from config.config import RAW_DATA_DIR


# ------------------------------------------------------
# Create claim folder
# ------------------------------------------------------
def create_claim_folder(claim_id: str) -> str:
    folder = os.path.join(RAW_DATA_DIR, claim_id)
    os.makedirs(folder, exist_ok=True)
    return folder


# ------------------------------------------------------
# Save uploaded files (simple, clean version)
# ------------------------------------------------------
def save_uploaded_files(uploaded_files, claim_id=None):
    """
    Saves uploaded files to the raw data directory.
    
    Always returns a list of dictionaries with structure:
    {
        "claim_id": ...,
        "original_name": ...,
        "saved_path": ...,
        "size": ...,
        "saved": True/False,
        "notes": ...
    }
    """
    results = []

    if not uploaded_files:
        return results

    # Generate claim ID if not provided
    if claim_id is None:
        claim_id = f"claim_{uuid.uuid4().hex[:8]}"

    claim_folder = create_claim_folder(claim_id)

    for file_obj in uploaded_files:
        # Get file name safely
        filename = Path(getattr(file_obj, "name", "uploaded_file")).name
        safe_filename = filename.replace("..", "").replace("/", "_")

        save_path = os.path.join(claim_folder, safe_filename)

        file_info = {
            "claim_id": claim_id,
            "original_name": filename,
            "saved_path": save_path,
            "size": 0,
            "saved": False,
            "notes": ""
        }

        try:
            # Write the file bytes
            with open(save_path, "wb") as f:
                if hasattr(file_obj, "read"):
                    f.write(file_obj.read())
                elif hasattr(file_obj, "getbuffer"):
                    f.write(file_obj.getbuffer())
                else:
                    file_info["notes"] = "Unsupported file object"
                    results.append(file_info)
                    continue

            # File saved successfully
            file_info["saved"] = True
            file_info["size"] = os.path.getsize(save_path)
            file_info["notes"] = "File saved successfully"

        except Exception as e:
            file_info["saved"] = False
            file_info["notes"] = str(e)

        results.append(file_info)

    return results

