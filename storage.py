import os
import uuid
from pathlib import Path
from fastapi import UploadFile

UPLOAD_DIR = Path("uploads")

def ensure_upload_dir():
    UPLOAD_DIR.mkdir(exist_ok=True)

def save_upload_file(file: UploadFile) -> dict:
    ensure_upload_dir()

    file_id = str(uuid.uuid4())
    extension = Path(file.filename).suffix
    filename = f"{file_id}{extension}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return {
        "file_id": file_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "stored_path": str(file_path)
    }

def get_file_path(file_id: str) -> Path | None:
    ensure_upload_dir()

    for file in UPLOAD_DIR.iterdir():
        if file.name.startswith(file_id):
            return file
    return None
