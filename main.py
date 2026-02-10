import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# OpenAI SDK
from openai import OpenAI

app = FastAPI(title="Mayu AI Backend", version="0.1.0")

# ===============================
# OpenAI config
# ===============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ===============================
# Health
# ===============================
@app.get("/health")
def health():
    return {"ok": True}


# ===============================
# Models
# ===============================
class RefineRequest(BaseModel):
    doctorText: str = ""
    attachmentsText: str = ""
    transcriptText: str = ""


class ParseHistoryRequest(BaseModel):
    text: str


# ===============================
# Endpoints
# ===============================
@app.post("/refine")
async def refine(payload: RefineRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no está configurada en Render")

    user_input = f"""
Eres un asistente clínico. Devuelve JSON válido con:
- soap: {{S,O,A,P}} (strings)
- summary: string
- rp: string

Texto del doctor:
{payload.doctorText}

Transcripción:
{payload.transcriptText}

Texto adjuntos:
{payload.attachmentsText}
""".strip()

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=user_input,
            text={"format": {"type": "json_object"}},
        )

        # En Responses API, lo más práctico aquí es:
        output_text = getattr(resp, "output_text", None)

        if not output_text:
            # fallback si output_text no viene por alguna razón
            output_text = json.dumps({
                "soap": {"S": "NR", "O": "NR", "A": "NR", "P": "NR"},
                "summary": "No output_text in response",
                "rp": "NR"
            })

        data = json.loads(output_text)
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    # Placeholder: aquí luego conectas OCR / PDF parsing
    return {"text": "NR"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Placeholder: aquí luego conectas Whisper / audio
    return {"text": "NR"}


@app.post("/parse_history")
async def parse_history(payload: ParseHistoryRequest):
    # Placeholder: luego puedes usar OpenAI aquí también
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
# DEBUG (no expone secretos)
# ===============================
@app.get("/debug/env")
def debug_env():
    return {
        "has_openai_key": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL
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
        return {"openai": getattr(r, "output_text", "OK")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")
