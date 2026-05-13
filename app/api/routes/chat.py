# app/api/routes/chat.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from rag.chain import answer_question, stream_answer
import json

router = APIRouter()

class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    question:     str
    chat_history: Optional[list[ChatMessage]] = []
    doc_id:       Optional[str] = None

class SourceItem(BaseModel):
    filename:    str
    chunk_index: int
    score:       float

class ChatResponse(BaseModel):
    answer:          str
    sources:         list[SourceItem]
    rewritten_query: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.chat_history
        ]
        result = answer_question(
            question=request.question,
            chat_history=history,
            doc_id=request.doc_id
        )
        return ChatResponse(
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
            rewritten_query=result["rewritten_query"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint — returns tokens as they are generated."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    history = [
        {"role": msg.role, "content": msg.content}
        for msg in request.chat_history
    ]

    async def generate():
        async for chunk in stream_answer(
            question=request.question,
            chat_history=history,
            doc_id=request.doc_id
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")