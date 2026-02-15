import os
import io
import json
import shutil
import tempfile
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from openai import OpenAI


# ===============================
# App
# ===============================
app = FastAPI(title="Mayu AI Backend", version="0.2.0")


# ===============================
# OpenAI config (ENV)
# ===============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_AUDIO_MODEL = os.getenv("OPENAI_AUDIO_MODEL", "gpt-4o-mini-transcribe")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ===============================
# Limits / OCR
# ===============================
MAX_EXTRACT_CHARS = int(os.getenv("MAX_EXTRACT_CHARS", "120000"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "80"))
OCR_DPI = int(os.getenv("OCR_DPI", "220"))
OCR_LANG = os.getenv("OCR_LANG", "spa+eng")

# En Docker/Render normalmente NO hace falta. En Mac puede ser útil.
# export POPPLER_PATH="/opt/homebrew/bin"
POPPLER_PATH = (os.getenv("POPPLER_PATH", "").strip() or None)


def _clip_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "\n\n[...TRUNCADO POR LÍMITE...]"


from typing import Optional


def _guess_ext(filename: str, content_type: Optional[str]) -> str:
    name = (filename or "").lower()
    ct = (content_type or "").lower()

    # PDF
    if name.endswith(".pdf") or "application/pdf" in ct:
        return "pdf"

    # Imágenes por extensión
    if any(name.endswith("." + x) for x in ["jpg", "jpeg", "png", "webp", "tif", "tiff"]):
        return name.split(".")[-1]

    # Imágenes por content-type
    if ct in ["image/jpeg", "image/png", "image/webp", "image/tiff"]:
        if ct == "image/jpeg":
            return "jpg"
        if ct == "image/png":
            return "png"
        if ct == "image/webp":
            return "webp"
        if ct == "image/tiff":
            return "tiff"

    # Fallback por extensión si existe
    if "." in name:
        return name.split(".")[-1]

    return ""

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
# /refine
# ===============================
@app.post("/refine")
async def refine(payload: RefineRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

    doctor_text = _clip_text(payload.doctorText, 20000)
    transcript_text = _clip_text(payload.transcriptText, 40000)
    attachments_text = _clip_text(payload.attachmentsText, 60000)

    prompt = f"""
Devuelve SOLO JSON válido con esta estructura exacta:
{{
  "soap": {{"S": "...", "O": "...", "A": "...", "P": "..."}},
  "summary": "...",
  "rp": "..."
}}

Reglas:
- Todo debe ser string.
- Sin texto fuera del JSON.
- En español.
- Si falta algo usa "NR".

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

        data = json.loads(resp.output_text or "{}")
        if not isinstance(data, dict):
            data = {}

        soap = data.get("soap")
        if not isinstance(soap, dict):
            soap = {}

        normalized_soap = {}
        for k in ["S", "O", "A", "P"]:
            v = soap.get(k, "")
            normalized_soap[k] = (v if isinstance(v, str) else str(v)).strip() or "NR"

        summary = data.get("summary", "NR")
        rp = data.get("rp", "NR")

        summary = (summary if isinstance(summary, str) else str(summary)).strip() or "NR"
        rp = (rp if isinstance(rp, str) else str(rp)).strip() or "NR"

        return {"soap": normalized_soap, "summary": summary, "rp": rp}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="OpenAI devolvió JSON inválido")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Refine error: {type(e).__name__}: {str(e)}",
        )


# ===============================
# /extract  (PDF + OCR fallback)
# debug=true para ver por qué falló (ya no más NR ciego)
# ===============================
@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    debug: bool = Query(False),
):
    ext = _guess_ext(file.filename or "file", file.content_type)

    # Guardar temporal (con suffix correcto)
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + (ext or "bin")) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    def _nr(debug_msg: str):
        return {"text": "NR", "debug": debug_msg} if debug else {"text": "NR"}

    try:
        extracted = ""

        # ================= PDF =================
        if ext == "pdf":
            pymupdf_err = None
            pypdf2_err = None
            ocr_err = None

            # 1) PyMuPDF (mejor para PDFs con texto real)
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(tmp_path)
                page_count = min(len(doc), MAX_PDF_PAGES)

                for i in range(page_count):
                    t = (doc[i].get_text("text") or "").strip()
                    if t:
                        extracted += t + "\n\n"
                    if len(extracted) >= MAX_EXTRACT_CHARS:
                        break

                doc.close()

            except Exception as e:
                extracted = ""
                pymupdf_err = f"PyMuPDF error: {type(e).__name__}: {str(e)}"

            # ✅ Si PyMuPDF devuelve demasiado poco, forzamos OCR (caso Render frecuente)
            if len(extracted.strip()) < 1500:
                extracted = ""

            # 2) PyPDF2 fallback
            if not extracted.strip():
                try:
                    from PyPDF2 import PdfReader

                    reader = PdfReader(tmp_path)
                    page_count = min(len(reader.pages), MAX_PDF_PAGES)

                    for i in range(page_count):
                        t = (reader.pages[i].extract_text() or "").strip()
                        if t:
                            extracted += t + "\n\n"
                        if len(extracted) >= MAX_EXTRACT_CHARS:
                            break

                except Exception as e:
                    extracted = ""
                    pypdf2_err = f"PyPDF2 error: {type(e).__name__}: {str(e)}"

            # ✅ Si sigue corto, OCR completo
            if len(extracted.strip()) < 1500:
                try:
                    from pdf2image import convert_from_path
                    import pytesseract

                    images = convert_from_path(
                        tmp_path,
                        dpi=OCR_DPI,
                        first_page=1,
                        last_page=MAX_PDF_PAGES,
                        poppler_path=POPPLER_PATH,  # None en Docker/Render suele estar bien
                    )

                    ocr_text = ""
                    for img in images:
                        t = (pytesseract.image_to_string(img, lang=OCR_LANG) or "").strip()
                        if t:
                            ocr_text += t + "\n\n"
                        if len(ocr_text) >= MAX_EXTRACT_CHARS:
                            break

                    if ocr_text.strip():
                        extracted = ocr_text

                except Exception as e:
                    ocr_err = f"OCR error: {type(e).__name__}: {str(e)}"

            final_text = _clip_text(extracted, MAX_EXTRACT_CHARS).strip()
            if not final_text:
                return _nr(
                    "No se extrajo texto (vacío) | "
                    f"{pymupdf_err or 'PyMuPDF: OK?'} | "
                    f"{pypdf2_err or 'PyPDF2: OK?'} | "
                    f"{ocr_err or 'OCR: OK?'} | "
                    f"POPPLER_PATH={POPPLER_PATH} | "
                    f"which(pdftoppm)={shutil.which('pdftoppm')} | "
                    f"which(tesseract)={shutil.which('tesseract')}"
                )

            if debug:
                return {
                    "text": final_text,
                    "debug": {
                        "ext": ext,
                        "pages_limit": MAX_PDF_PAGES,
                        "chars_limit": MAX_EXTRACT_CHARS,
                        "pymupdf_err": pymupdf_err,
                        "pypdf2_err": pypdf2_err,
                        "ocr_err": ocr_err,
                        "ocr_lang": OCR_LANG,
                        "ocr_dpi": OCR_DPI,
                        "poppler_path": POPPLER_PATH,
                        "which_pdftoppm": shutil.which("pdftoppm"),
                        "which_tesseract": shutil.which("tesseract"),
                        "text_len": len(final_text),
                    },
                }

            return {"text": final_text}

        # ================= IMAGES =================
        if ext in ["jpg", "jpeg", "png", "webp", "tif", "tiff"]:
            try:
                from PIL import Image
                import pytesseract

                img = Image.open(tmp_path)
                extracted_img = (pytesseract.image_to_string(img, lang=OCR_LANG) or "").strip()
                final_img = _clip_text(extracted_img, MAX_EXTRACT_CHARS).strip()

                if not final_img:
                    return _nr("OCR de imagen devolvió vacío.")

                if debug:
                    return {
                        "text": final_img,
                        "debug": {
                            "ext": ext,
                            "ocr_lang": OCR_LANG,
                            "ocr_dpi": OCR_DPI,
                            "text_len": len(final_img),
                        },
                    }

                return {"text": final_img}

            except Exception as e:
                return _nr(f"Image OCR error: {type(e).__name__}: {str(e)}")

        # Otros
        return _nr(
            f"Extensión no soportada: {ext} (filename={file.filename}, ct={file.content_type})"
        )

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ===============================
# /transcribe (audio -> texto)
# ===============================
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return {"text": "NR"}
    except Exception:
        return {"text": "NR"}

    f = io.BytesIO(audio_bytes)
    f.name = file.filename or "audio.m4a"
    content_type = file.content_type or "application/octet-stream"

    try:
        r = client.audio.transcriptions.create(
            model=OPENAI_AUDIO_MODEL,
            file=(f.name, f, content_type),
        )
        text = (getattr(r, "text", "") or "").strip()
        return {"text": text or "NR"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcribe error: {type(e).__name__}: {str(e)}",
        )


# ===============================
# /parse_history (stub)
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
# Debug endpoints
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
        "poppler_path_env": POPPLER_PATH,
        "which_pdftoppm": shutil.which("pdftoppm"),
        "which_tesseract": shutil.which("tesseract"),
    }


@app.get("/debug/openai")
def debug_openai():
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing")
    try:
        r = client.responses.create(model=OPENAI_MODEL, input="Responde solo: OK")
        return {"openai": (r.output_text or "").strip()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI error: {type(e).__name__}: {str(e)}",
        )


@app.get("/debug/system")
def debug_system():
    try:
        t = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        p = subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True)

        t_out = ((t.stdout or "") + (t.stderr or ""))[:400]
        p_out = ((p.stdout or "") + (p.stderr or ""))[:400]

        return {
            "tesseract_installed": t.returncode == 0,
            "poppler_installed": p.returncode == 0,
            "tesseract_output": t_out,
            "poppler_output": p_out,
            "tesseract_path": shutil.which("tesseract"),
            "pdftoppm_path": shutil.which("pdftoppm"),
        }
    except FileNotFoundError as e:
        return {"error": f"Binario no encontrado: {str(e)}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}
