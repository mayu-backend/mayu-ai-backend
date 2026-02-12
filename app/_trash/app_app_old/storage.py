# app/storage.py

from pathlib import Path
import uuid
import shutil

# Carpeta donde se guardan los archivos subidos
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_upload_file(file, filename: str) -> dict:
    """
    Guarda archivo en carpeta uploads
    Devuelve file_id y metadata
    """
    file_id = str(uuid.uuid4())

    # limpiar nombre
    safe_name = filename.replace("/", "_").replace("\\", "_")

    # ruta final
    path = UPLOAD_DIR / f"{file_id}__{safe_name}"

    with path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "file_id": file_id,
        "filename": safe_name,
        "path": str(path)
    }


def resolve_file_path(file_id: str) -> Path:
    """
    Busca archivo por file_id
    """
    matches = list(UPLOAD_DIR.glob(f"{file_id}__*"))
    if not matches:
        raise FileNotFoundError("file_id not found")

    return matches[0]
