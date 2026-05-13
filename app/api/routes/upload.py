# app/api/routes/upload.py
import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag.ingestion import ingest_document, delete_document
from app.config import UPLOAD_DIR

router = APIRouter()

class UploadResponse(BaseModel):
    message:      str
    filename:     str
    doc_id:       str
    total_pages:  int
    total_chunks: int

class DeleteResponse(BaseModel):
    message: str
    doc_id:  str

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Save file temporarily
    file_id   = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.pdf"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Ingest into Pinecone
        result = ingest_document(
            file_path=str(file_path),
            filename=file.filename
        )
        return UploadResponse(
            message="Document uploaded and indexed successfully",
            **result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_doc(doc_id: str):
    try:
        delete_document(doc_id)
        return DeleteResponse(
            message="Document deleted successfully",
            doc_id=doc_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))