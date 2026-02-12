import os
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    UploadResponse,
    ExtractTextRequest, ExtractTextResponse,
    TranscribeRequest, TranscribeResponse,
    RefineRequest, RefineResponse
)

# -----------------------------
# Config
# -----------------------------
APP_NAME = "Mayu AI Backend"
DATA_DIR = Path(os.getenv("MAYU_DATA_DIR", "./data")).resolve()
UPLOADS_DIR = DATA_DIR / "uploads"
INDEX_FILE = DATA_DIR / "index.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # el que ya usas

# -----------------------------
# Init folders
# -----------------------------
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
if not INDEX_FILE.exists():
    INDEX_FILE.write_text(json.dumps({"files": {}}, ensure_ascii=False, indent=2), encoding="utf-8")

# -----------------------------
# OpenAI client (opcional)
# -----------------------------
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None

# -----------------------------
# App
# -----------------------------
app = FastAPI(title=APP_NAME, version="1.0.0")

# CORS (ajusta dominios si quieres)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod pon tu dominio/app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers: index
# -----------------------------
def _load_index() -> Dict[str, Any]:
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"files": {}}

def _save_index(idx: Dict[str, Any]) -> None:
    INDEX_FILE.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

def _guess_ext(filename: str, content_type: str) -> str:
    name = (filename or "").lower()
    if "." in name:
        return "." + name.split(".")[-1]
    # fallback por content-type
    if content_type == "application/pdf":
        return ".pdf"
    if content_type in ("image/jpeg", "image/jpg"):
        return ".jpg"
    if content_type == "image/png":
        return ".png"
    if content_type.startswith("audio/"):
        return ".audio"
    return ".bin"

def _get_file_meta(file_id: str) -> Dict[str, Any]:
    idx = _load_index()
    meta = idx.get("files", {}).get(file_id)
    if not meta:
        raise HTTPException(status_code=404, detail="file_id no encontrado")
    return meta

def _get_file_path(file_id: str) -> Path:
    meta = _get_file_meta(file_id)
    p = Path(meta["path"])
    if not p.exists():
        raise HTTPException(status_code=404, detail="archivo no existe en disco")
    return p

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "name": APP_NAME}

# -----------------------------
# A) /upload
# -----------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No se recibió archivo")

    file_id = uuid.uuid4().hex
    content_type = file.content_type or "application/octet-stream"
    ext = _guess_ext(file.filename or "", content_type)

    safe_name = (file.filename or f"upload{ext}").replace("/", "_").replace("\\", "_")
    out_path = UPLOADS_DIR / f"{file_id}{ext}"

    try:
        data = await file.read()
        out_path.write_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando archivo: {str(e)}")

    idx = _load_index()
    idx["files"][file_id] = {
        "file_id": file_id,
        "filename": safe_name,
        "content_type": content_type,
        "path": str(out_path),
    }
    _save_index(idx)

    return UploadResponse(file_id=file_id, filename=safe_name, content_type=content_type)

# -----------------------------
# B) /extract-text
# -----------------------------
@app.post("/extract-text", response_model=ExtractTextResponse)
def extract_text(payload: ExtractTextRequest):
    p = _get_file_path(payload.file_id)
    meta = _get_file_meta(payload.file_id)
    ct = (meta.get("content_type") or "").lower()
    name = (meta.get("filename") or "").lower()

    # 1) TXT directo
    if ct.startswith("text/") or name.endswith(".txt"):
        try:
            return ExtractTextResponse(text=p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return ExtractTextResponse(text=p.read_text(errors="ignore"))

    # 2) PDF -> PyMuPDF (fitz)
    if ct == "application/pdf" or name.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
        except Exception:
            raise HTTPException(
                status_code=501,
                detail="Falta PyMuPDF. Instala: pip install pymupdf"
            )

        try:
            doc = fitz.open(str(p))
            chunks = []
            for i in range(len(doc)):
                page = doc[i]
                chunks.append(page.get_text("text"))
            text = "\n".join([c.strip() for c in chunks if c.strip()])
            return ExtractTextResponse(text=text.strip())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extrayendo texto PDF: {str(e)}")

    # 3) Imágenes -> OCR (pytesseract)
    if ct.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        try:
            from PIL import Image
            import pytesseract
        except Exception:
            raise HTTPException(
                status_code=501,
                detail="Falta OCR. Instala: pip install pillow pytesseract (y tener tesseract instalado en el sistema)."
            )
        try:
            img = Image.open(str(p))
            text = pytesseract.image_to_string(img, lang="spa+eng")
            return ExtractTextResponse(text=text.strip())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error OCR: {str(e)}")

    # 4) Fallback: binario sin extractor
    raise HTTPException(
        status_code=415,
        detail=f"No soportado para extracción de texto: {ct or 'unknown'}"
    )

# -----------------------------
# C) /transcribe (audio)
# -----------------------------
@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(payload: TranscribeRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada o OpenAI client no disponible")

    p = _get_file_path(payload.file_id)
    meta = _get_file_meta(payload.file_id)
    ct = (meta.get("content_type") or "").lower()

    if not (ct.startswith("audio/") or p.suffix.lower() in [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"]):
        # no bloqueamos del todo, pero avisamos
        pass

    try:
        # OpenAI Audio Transcriptions (SDK moderno)
        with open(p, "rb") as f:
            result = client.audio.transcriptions.create(
                model=os.getenv("OPENAI_AUDIO_MODEL", "gpt-4o-mini-transcribe"),
                file=f,
            )
        text = getattr(result, "text", "") or ""
        return TranscribeResponse(text=text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribiendo audio: {str(e)}")

# -----------------------------
# D) /refine (SOAP / resumen / RP / unified)
# -----------------------------
@app.post("/refine", response_model=RefineResponse)
def refine(payload: RefineRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada o OpenAI client no disponible")

    doctor_text = (payload.doctorText or "").strip()
    attachments_text = (payload.attachmentsText or "").strip()
    transcript_text = (payload.transcriptText or "").strip()

    unified_input = "\n\n".join([
        "=== DOCTOR TEXT ===\n" + doctor_text if doctor_text else "",
        "=== ATTACHMENTS TEXT ===\n" + attachments_text if attachments_text else "",
        "=== TRANSCRIPT TEXT ===\n" + transcript_text if transcript_text else "",
    ]).strip()

    if not unified_input:
        raise HTTPException(status_code=400, detail="No hay texto para refinar")

    system = (
        "Eres un asistente clínico para documentación médica. "
        "Devuelve SIEMPRE JSON válido con estas llaves: soap, resumen, rp, unified. "
        "No inventes datos: si falta algo, pon 'NR'. "
        "soap debe ser estilo SOAP (S/O/A/P). rp es receta/plan terapéutico en bullets."
    )

    user = (
        "Refina y unifica la información clínica.\n\n"
        f"{unified_input}\n\n"
        "Devuelve JSON."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )

        content = resp.choices[0].message.content or ""
        # Intentar parsear JSON
        try:
            data = json.loads(content)
        except Exception:
            # fallback: envolver texto
            data = {
                "soap": content,
                "resumen": "",
                "rp": "",
                "unified": unified_input
            }

        return RefineResponse(
            soap=str(data.get("soap", "") or ""),
            resumen=str(data.get("resumen", "") or ""),
            rp=str(data.get("rp", "") or ""),
            unified=str(data.get("unified", "") or unified_input),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en refine: {str(e)}")
