from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="Mayu AI Backend", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True}

class RefineRequest(BaseModel):
    doctorText: str = ""
    attachmentsText: str = ""
    transcriptText: str = ""

@app.post("/refine")
async def refine(payload: RefineRequest):
    return {
        "soap": {"S": "NR", "O": "NR", "A": "NR", "P": "NR"},
        "summary": "NR",
        "rp": "NR"
    }

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
