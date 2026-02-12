import os
import io
import json
import tempfile
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# ===============================
# App
# ===============================
app = FastAPI(title="Mayu AI Backend", version="0.2.0")

# ===============================
# OpenAI config (ENV only)
# ===============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Model recomendado para audio (si no lo defines, intentamos uno razonable)
OPENAI_AUDIO_MODEL = os.getenv("OPENAI_AUDIO_MODEL", "gpt-4o-mini-transcribe")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ===============================
# Helpers: limits (PDF largos)
# ===============================
MAX_EXTRACT_CHARS = int(os.getenv("MAX_EXTRACT_CHARS", "120000"))  # salida max de /extract
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "80"))              # limita páginas por seguridad/costo
OCR_DPI = int(os.getenv("OCR_DPI", "220"))                         # calidad OCR (sube si sale mal)
OCR_LANG = os.getenv("OCR_LANG", "spa+eng")                        # tesseract lang (spa+eng)


def _clip_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    # corte “amable”
    return s[:max_chars].rstrip() + "\n\n[...TRUNCADO POR LÍMITE...]"


# ===============================
# Health
# ===============================
@app.get("/health")
def health():
    return {"ok": True}


# ===============================
# Schemas
# ===============================
class RefineRequest(BaseModel):
    doctorText: str = ""
    attachmentsText: str = ""
    transcriptText: str = ""


class ParseHistoryRequest(BaseModel):
    text: str = ""


# ===============================
# Endpoint: /refine
# ===============================
@app.post("/refine")
async def refine(payload: RefineRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no está configurada")

    # ✅ Recorte para evitar reventar tokens en PDFs muy largos
    doctor_text = _clip_text(payload.doctorText, 20000)
    transcript_text = _clip_text(payload.transcriptText, 40000)
    attachments_text = _clip_text(payload.attachmentsText, 60000)

    # ✅ Mejor como prompt “limpio” + JSON estricto
    prompt = f"""
Eres un asistente clínico. Devuelve SOLO JSON válido con esta estructura exacta:
{{
  "soap": {{"S": "...", "O": "...", "A": "...", "P": "..."}},
  "summary": "...",
  "rp": "..."
}}

Reglas:
- "S","O","A","P","summary","rp" deben ser strings.
- No incluyas texto fuera del JSON.
- Sé conciso, clínico y en español.
- Si faltan datos, usa "NR".

Texto del doctor:
{doctor_text}

Transcripción:
{transcript_text}

Adjuntos:
{attachments_text}
""".strip()

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )

        data = json.loads(resp.output_text)

        if not isinstance(data, dict) or "soap" not in data:
            raise ValueError("JSON inválido o sin campo 'soap'")

        # Validación mínima de llaves SOAP
        soap = data.get("soap", {})
        for k in ["S", "O", "A", "P"]:
            if k not in soap:
                soap[k] = "NR"
        data["soap"] = soap

        # Garantiza strings
        for k in ["summary", "rp"]:
            if k not in data or not isinstance(data[k], str):
                data[k] = "NR"

        return data

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="OpenAI devolvió un JSON inválido")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")


# ===============================
# Endpoint: /extract  (PDF + OCR fallback)
# ===============================
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """
    Devuelve: {"text": "..."} siempre.
    Estrategia:
      1) PDF con texto: extrae por páginas (PyMuPDF si está, fallback PyPDF2)
      2) Si sale vacío -> OCR por páginas (pdf2image + tesseract)
      3) Imágenes directas -> OCR (PIL + tesseract)
    """

    filename = (file.filename or "file").lower()
    ext = filename.split(".")[-1] if "." in filename else ""

    # guardamos temporal para librerías OCR/PDF
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        text_out = ""

        # -------------------------
        # PDF
        # -------------------------
        if ext == "pdf":
            # 1) Intento PyMuPDF (mejor para PDFs largos y texto real)
            extracted = ""
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(tmp_path)
                page_count = min(len(doc), MAX_PDF_PAGES)

                for i in range(page_count):
                    pg = doc[i]
                    t = (pg.get_text("text") or "").strip()
                    if t:
                        extracted += t + "\n\n"
                    if len(extracted) >= MAX_EXTRACT_CHARS:
                        break

                doc.close()
            except Exception:
                extracted = ""

            # 2) Fallback: PyPDF2 si no hay PyMuPDF
            if not extracted.strip():
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(tmp_path)
                    page_count = min(len(reader.pages), MAX_PDF_PAGES)
                    for i in range(page_count):
                        pg = reader.pages[i]
                        t = (pg.extract_text() or "").strip()
                        if t:
                            extracted += t + "\n\n"
                        if len(extracted) >= MAX_EXTRACT_CHARS:
                            break
                except Exception:
                    extracted = ""

            # 3) Si sigue vacío -> OCR
            if not extracted.strip():
                try:
                    from pdf2image import convert_from_path
                    import pytesseract

                    images = convert_from_path(tmp_path, dpi=OCR_DPI, first_page=1, last_page=MAX_PDF_PAGES)
                    for img in images:
                        t = pytesseract.image_to_string(img, lang=OCR_LANG) or ""
                        t = t.strip()
                        if t:
                            extracted += t + "\n\n"
                        if len(extracted) >= MAX_EXTRACT_CHARS:
                            break
                except Exception:
                    extracted = ""

            text_out = _clip_text(extracted, MAX_EXTRACT_CHARS) or "NR"
            return {"text": text_out}

        # -------------------------
        # IMÁGENES -> OCR
        # -------------------------
        if ext in ["jpg", "jpeg", "png", "webp", "tif", "tiff"]:
            try:
                from PIL import Image
                import pytesseract

                img = Image.open(tmp_path)
                extracted = pytesseract.image_to_string(img, lang=OCR_LANG) or ""
                text_out = extracted.strip() or "NR"
                return {"text": _clip_text(text_out, MAX_EXTRACT_CHARS)}
            except Exception:
                return {"text": "NR"}

        # -------------------------
        # Otros -> NR
        # -------------------------
        return {"text": "NR"}

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ===============================
# Endpoint: /transcribe  (audio -> texto)
# ===============================
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no está configurada")

    # Leemos bytes del audio
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return {"text": "NR"}
    except Exception:
        return {"text": "NR"}

    # Nombre y mimetype
    filename = file.filename or "audio.m4a"
    content_type = file.content_type or "application/octet-stream"

    # OpenAI espera un “file-like” (io.BytesIO) con name
    f = io.BytesIO(audio_bytes)
    f.name = filename  # importante para algunos parsers

    try:
        # Nota: según versión del SDK, esto puede ser:
        # client.audio.transcriptions.create(...)
        # Si tu SDK difiere, me dices y lo ajusto.
        r = client.audio.transcriptions.create(
            model=OPENAI_AUDIO_MODEL,
            file=(filename, f, content_type),
        )

        # r.text suele venir en transcriptions
        text = getattr(r, "text", None) or ""
        text = text.strip()
        return {"text": text if text else "NR"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcribe error: {str(e)}")


# ===============================
# Endpoint: /parse_history
# (lo dejas como lo tienes por ahora)
# ===============================
@app.post("/parse_history")
async def parse_history(payload: ParseHistoryRequest):
    return {
        "pathologic": None,
        "family": None,
        "surgical": None,
        "allergies": None,
        "currentMeds": None,
        "supplements": None,
        "summary": None,
        "structured": None,
    }


# ===============================
# Debug (safe)
# ===============================
@app.get("/debug/env")
def debug_env():
    return {
        "has_openai_key": bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL,
        "openai_audio_model": OPENAI_AUDIO_MODEL,
        "max_extract_chars": MAX_EXTRACT_CHARS,
        "max_pdf_pages": MAX_PDF_PAGES,
        "ocr_dpi": OCR_DPI,
        "ocr_lang": OCR_LANG,
    }


@app.get("/debug/openai")
def debug_openai():
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing in environment")
    try:
        r = client.responses.create(
            model=OPENAI_MODEL,
            input="Responde solo: OK",
        )
        return {"openai": (r.output_text or "").strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")
import subprocess

@app.get("/debug/system")
def debug_system():
    try:
        tesseract = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True
        )
        poppler = subprocess.run(
            ["pdftoppm", "-v"],
            capture_output=True,
            text=True
        )

        return {
            "tesseract_installed": tesseract.returncode == 0,
            "tesseract_output": tesseract.stdout[:200],
            "poppler_installed": poppler.returncode == 0,
            "poppler_output": poppler.stdout[:200],
        }

    except Exception as e:
        return {"error": str(e)}
