# app/schemas.py

from pydantic import BaseModel


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    content_type: str


class FileIdRequest(BaseModel):
    file_id: str


class ExtractTextResponse(BaseModel):
    text: str


class TranscribeResponse(BaseModel):
    text: str


class RefineRequest(BaseModel):
    doctorText: str = ""
    attachmentsText: str = ""
    transcriptText: str = ""


class RefineResponse(BaseModel):
    soap: str = ""
    resumen: str = ""
    rp: str = ""
    unified: str = ""
