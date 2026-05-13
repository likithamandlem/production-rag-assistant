# app/main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from app.api.routes import upload, chat

app = FastAPI(
    title="Production RAG Assistant",
    description="Advanced RAG system with Pinecone + Groq + Reranking",
    version="1.0.0"
)

app.include_router(upload.router, prefix="/api/v1", tags=["Documents"])
app.include_router(chat.router,   prefix="/api/v1", tags=["Chat"])

@app.get("/")
def root():
    return {
        "message": "Production RAG Assistant is running",
        "docs":    "/docs",
        "health":  "/health"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}