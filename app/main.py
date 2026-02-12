# app/main.py

import os
import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from openai import OpenAI

from storage import save_upload_file, resolve_file_path
from schemas import (
    UploadResponse,
    FileIdRequest,
    FileIdsRequest,
    ExtractTextResponse,
    ExtractManyResponse,
    TranscribeResponse,
    RefineRequest,
    RefineResponse,
    AutoRefineRequest,
    AutoRefineResponse,
)

# ---------------- OpenAI ----------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# ---------------- App ----------------

app = FastAPI(title="Mayu AI Backend", version="0.2.0")

# ---------------- CORS ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Helpers ----------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
TEXT_EXTS = {".txt", ".md"}
PDF_EXTS = {".pdf"}
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".ogg", ".webm"}


def _ext(p: Path) -> str:
    return p.suffix.lower()


def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return p.read_bytes().decode("utf-8", errors="ignore")


def _ocr_image_file(p: Path) -> str:
    img = Image.open(p).convert("RGB")
    # Si no tienes spa instalado en tesseract, cambia a "eng"
    return pytesseract.image_to_string(img, lang="spa")


def _extract_pdf_text(p: Path) -> str:
    doc = fitz.open(p)
    chunks: List[str] = []
    for i in range(len(doc)):
        page = doc[i]
        t = page.get_text("text").strip()
        if t:
            chunks.append(t)
    doc.close()
    return "\n\n".join(chunks).strip()


def _ocr_pdf_if_needed(p: Path, max_pages: int = 5) -> str:
    """
    Si el PDF viene escaneado y no trae texto, renderiza páginas y hace OCR.
    Limita páginas para evitar carga pesada.
    """
    doc = fitz.open(p)
    chunks: List[str] = []
    pages = min(len(doc), max_pages)

    for i in range(pages):
        page = doc[i]
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="spa").strip()
        if text:
            chunks.append(text)

    doc.close()
    return "\n\n".join(chunks).strip()


def extract_text_from_any(path: Path) -> str:
    ext = _ext(path)

    if ext in TEXT_EXTS:
        return _read_text_file(path).strip()

    if ext in IMAGE_EXTS:
        return _ocr_image_file(path).strip()

    if ext in PDF_EXTS:
        txt = _extract_pdf_text(path)
        if txt:
            return txt
        return _ocr_pdf_if_needed(path).strip()

    raise HTTPException(status_code=400, detail=f"Tipo de archivo no soportado: {ext}")


def transcribe_audio_with_openai(path: Path) -> str:
    """
    Transcribe audio usando OpenAI (Whisper).
    Requiere OPENAI_API_KEY seteada en ENV.
    """
    ext = _ext(path)
    if ext not in AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"Archivo no es audio soportado: {ext}")

    try:
        with path.open("rb") as f:
            result = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f,
                response_format="text",
            )

        # según SDK puede ser str o un objeto con .text
        if isinstance(result, str):
            return result.strip()
        text = getattr(result, "text", "")
        return (text or "").strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribiendo audio: {str(e)}")


def _truncate(text: str, max_chars: int = 45000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCADO POR LÍMITE DE TAMAÑO]"


def refine_with_openai(unified_text: str) -> dict:
    """
    Devuelve dict con formato:
      {
        "soap": {"S": "...", "O": "...", "A": "...", "P": "..."},
        "summary": "...",
        "rp": "..."
      }
    """
    unified_text = _truncate(unified_text)

    system = (
        "Eres un asistente clínico. Genera un SOAP, un resumen clínico breve y recomendaciones.\n"
        "Devuelve SOLO JSON válido con llaves EXACTAS:\n"
        '{ "soap": {"S":"", "O":"", "A":"", "P":""}, "summary":"", "rp":"" }\n'
        "No incluyas texto fuera del JSON."
    )

    user = (
        "Usa este contenido para generar SOAP + summary + rp:\n\n"
        f"{unified_text}"
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)

        soap = data.get("soap") or {}
        out = {
            "soap": {
                "S": (soap.get("S") or "").strip() or "NR",
                "O": (soap.get("O") or "").strip() or "NR",
                "A": (soap.get("A") or "").strip() or "NR",
                "P": (soap.get("P") or "").strip() or "NR",
            },
            "summary": (data.get("summary") or "").strip() or "NR",
            "rp": (data.get("rp") or "").strip() or "NR",
        }
        return out

    except Exception as e:
        return {
            "soap": {"S": "NR", "O": "NR", "A": f"Error IA: {str(e)}", "P": "NR"},
            "summary": "NR",
            "rp": "NR",
        }


# ---------------- Endpoints ----------------

@app.get("/health")
def health():
    return {"ok": True}


# ---------- UPLOAD ----------

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename requerido")

    meta = save_upload_file(file, file.filename)

    return UploadResponse(
        file_id=meta["file_id"],
        filename=meta["filename"],
        content_type=file.content_type or "application/octet-stream",
    )


# ---------- EXTRACT ONE ----------

@app.post("/extract-text", response_model=ExtractTextResponse)
async def extract_text(request: FileIdRequest):
    try:
        path = resolve_file_path(request.file_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="file_id not found")

    text = extract_text_from_any(path)
    return ExtractTextResponse(text=text)


# ---------- EXTRACT MANY ----------

@app.post("/extract-many", response_model=ExtractManyResponse)
async def extract_many(request: FileIdsRequest):
    combined_parts: List[str] = []

    for fid in request.file_ids:
        try:
            path = resolve_file_path(fid)
            text = extract_text_from_any(path)
            if text:
                combined_parts.append(f"--- Documento {fid} ---\n{text}")
        except Exception as e:
            combined_parts.append(f"--- Error {fid}: {str(e)} ---")

    combined = "\n\n".join(combined_parts).strip()
    return ExtractManyResponse(text=combined)


# ---------- TRANSCRIBE ----------

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: FileIdRequest):
    try:
        path = resolve_file_path(request.file_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="file_id not found")

    text = transcribe_audio_with_openai(path)
    return TranscribeResponse(text=text)


# ---------- REFINE (REAL con OpenAI) ----------

@app.post("/refine", response_model=RefineResponse)
async def refine(request: RefineRequest):
    unified = (
        "DOCTOR:\n"
        f"{request.doctorText}\n\n"
        "ADJUNTOS:\n"
        f"{request.attachmentsText}\n\n"
        "AUDIO:\n"
        f"{request.transcriptText}\n"
    )

    result = refine_with_openai(unified)

    # Mantener response_model (schemas.py) compatible:
    # soap debe ser objeto con S/O/A/P, resumen = summary
    return RefineResponse(
        soap=result["soap"],
        resumen=result["summary"],
        rp=result["rp"],
        unified=unified,
    )


# ---------- AUTO REFINE (extrae + transcribe + refine real) ----------

@app.post("/auto-refine", response_model=AutoRefineResponse)
async def auto_refine(request: AutoRefineRequest):
    # 1) Extract files
    attachments_parts: List[str] = []
    for fid in request.file_ids:
        try:
            path = resolve_file_path(fid)
            text = extract_text_from_any(path)
            if text:
                attachments_parts.append(f"--- Documento {fid} ---\n{text}")
        except Exception as e:
            attachments_parts.append(f"--- Error {fid}: {str(e)} ---")

    attachments_text = "\n\n".join(attachments_parts).strip()

    # 2) Transcribe audios
    transcript_parts: List[str] = []
    for fid in request.audio_file_ids:
        try:
            path = resolve_file_path(fid)
            t = transcribe_audio_with_openai(path)
            transcript_parts.append(f"--- Audio {fid} ---\n{t}")
        except Exception as e:
            transcript_parts.append(f"--- Error audio {fid}: {str(e)} ---")

    transcript_text = "\n\n".join(transcript_parts).strip()

    # 3) Unified
    unified = (
        "DOCTOR:\n"
        f"{request.doctorText}\n\n"
        "ADJUNTOS:\n"
        f"{attachments_text}\n\n"
        "AUDIO:\n"
        f"{transcript_text}\n"
    )

    result = refine_with_openai(unified)

    return AutoRefineResponse(
        soap=result["soap"],
        resumen=result["summary"],
        rp=result["rp"],
        unified=unified,
    )
