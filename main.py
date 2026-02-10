import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from openai import OpenAI

app = FastAPI(title="Mayu AI Backend", version="0.1.0")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


@app.get("/health")
def health():
    return {"ok": True}


# ---------- DEBUG ----------
@app.get("/debug/env")
def debug_env():
    return {
        "has_openai_key": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
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
        return {"openai": r.output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")


# ---------- REFINE ----------
class RefineRequest(BaseModel):
    doctorText: str = ""
    attachmentsText: str = ""
    transcriptText: str = ""


@app.post("/refine")
async def refine(payload: RefineRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no está configurada en Render")

    user_input = f"""
Eres un asistente clínico. Devuelve SOLO JSON válido con esta estructura:
{{
  "soap": {{"S": "...", "O": "...", "A": "...", "P": "..."}},
  "summary": "...",
  "rp": "..."
}}

Texto del doctor:
{payload.doctorText}

Transcripción:
{payload.transcriptText}

Adjuntos:
{payload.attachmentsText}
""".strip()

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=user_input,
        text={"format": {"type": "json_object"}},
    )

    try:
        return json.loads(resp.output_text)
    except Exception:
        return {
            "soap": {"S": "NR", "O": "NR", "A": "NR", "P": "NR"},
            "summary": resp.output_text,
            "rp": "NR",
        }


# ---------- PLACEHOLDERS ----------
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    return {"text": "NR"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    return {"text": "NR"}


class ParseHistoryRequest(BaseModel):
    text: str


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
