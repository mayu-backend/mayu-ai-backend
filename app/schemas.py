from pydantic import BaseModel
from typing import List


# ============================
# UPLOAD
# ============================

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    content_type: str


class FileIdRequest(BaseModel):
    file_id: str


class FileIdsRequest(BaseModel):
    file_ids: List[str]


# ============================
# EXTRACT
# ============================

class ExtractTextResponse(BaseModel):
    text: str


class ExtractManyResponse(BaseModel):
    text: str


# ============================
# TRANSCRIBE
# ============================

class TranscribeResponse(BaseModel):
    text: str


# ============================
# REFINE (COMPATIBLE CON SWIFT)
# ============================

class RefineRequest(BaseModel):
    doctorText: str = ""
    attachmentsText: str = ""
    transcriptText: str = ""


class SOAP(BaseModel):
    S: str = ""
    O: str = ""
    A: str = ""
    P: str = ""


class RefineResponse(BaseModel):
    soap: SOAP
    summary: str = ""
    rp: str = ""


# ============================
# AUTO REFINE
# ============================

class AutoRefineRequest(BaseModel):
    doctorText: str = ""
    file_ids: List[str] = []
    audio_file_ids: List[str] = []


class AutoRefineResponse(BaseModel):
    soap: SOAP
    summary: str = ""
    rp: str = ""
