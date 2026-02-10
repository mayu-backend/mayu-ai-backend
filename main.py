import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Mayu AI Backend", version="0.2.0")

# ===============================
# OpenAI config (ENV only)
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
# Schemas
# ===============================
class RefineRequest(BaseModel):
    doctorText: str = ""
    attachmentsText: str = ""
    transcriptText: str = ""


class ParseHistoryRequest(BaseModel):
    text: str = ""


# ===============================
# Endpoints
# ===============================
@app.post("/refine")
async def refine(payload: RefineRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no está configurada")

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

Texto del doctor:
{payload.doctorText}

Transcripción:
{payload.transcriptText}

Adjuntos:
{payload.attachmentsText}
""".strip()

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )

        # output_text debería ser JSON string
        data = json.loads(resp.output_text)

        # Validación mínima del shape
        if not isinstance(data, dict) or "soap" not in data:
            raise ValueError("JSON inválido o sin campo 'soap'")

        return data

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="OpenAI devolvió un JSON inválido")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    return {"text": "NR"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    return {"text": "NR"}


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
