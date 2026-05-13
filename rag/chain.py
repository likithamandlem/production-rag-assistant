# rag/chain.py
from groq import Groq
from rag.retriever import retrieve
from rag.reranker import rerank
from app.config import GROQ_API_KEY, LLM_MODEL, TEMPERATURE
import json

client = Groq(api_key=GROQ_API_KEY)

def rewrite_query(question: str, chat_history: list[dict] = []) -> str:
    """Rewrite user query to improve retrieval quality."""
    history_text = ""
    for msg in chat_history[-4:]:
        history_text += f"{msg.get('role','')}: {msg.get('content','')}\n"

    prompt = f"""You are a query rewriting expert for a RAG system.
Rewrite the user's question to improve document retrieval.

Rules:
- Expand short or vague queries into detailed ones
- Include relevant synonyms and related terms
- Use context from chat history for follow-up questions
- Return ONLY the rewritten query — nothing else
- Keep it under 100 words

Chat History:
{history_text}

Original Question: {question}

Rewritten Question:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def build_context(chunks: list[dict]) -> str:
    """Build context string from reranked chunks."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\nSource {i+1} [{chunk['filename']}]:\n{chunk['text']}\n"
    return context

def build_sources(chunks: list[dict]) -> list[dict]:
    """Build source citations."""
    sources = []
    seen = set()
    for chunk in chunks:
        key = f"{chunk['filename']}_{chunk['chunk_index']}"
        if key not in seen:
            sources.append({
                "filename":    chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "score":       round(chunk.get("rerank_score", chunk["score"]), 4)
            })
            seen.add(key)
    return sources

def answer_question(
    question: str,
    chat_history: list[dict] = [],
    doc_id: str = None
) -> dict:
    """Full RAG pipeline with query rewriting."""

    # Step 1: Rewrite query
    rewritten_query = rewrite_query(question, chat_history)

    # Step 2: Retrieve
    chunks = retrieve(query=rewritten_query, doc_id=doc_id)

    if not chunks:
        return {
            "answer":          "No relevant documents found. Please upload documents first.",
            "sources":         [],
            "rewritten_query": rewritten_query
        }

    # Step 3: Rerank
    reranked_chunks = rerank(query=rewritten_query, chunks=chunks)

    # Step 4: Build context
    context = build_context(reranked_chunks)

    # Step 5: Build history
    history_text = ""
    for msg in chat_history[-6:]:
        history_text += f"{msg.get('role','')}: {msg.get('content','')}\n"

    # Step 6: Build prompt
    system_prompt = """You are a helpful AI assistant that answers questions
based on the provided document context.

Rules:
- Answer ONLY based on the context provided
- If the answer is not in the context, say "I could not find this information in the uploaded documents"
- Always cite which Source you used (Source 1, Source 2, etc.)
- Be concise and clear
- Never hallucinate information"""

    user_prompt = f"""Context:
{context}

Chat History:
{history_text}

Original Question: {question}

Answer:"""

    # Step 7: Call Groq
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=1024
    )

    answer  = response.choices[0].message.content
    sources = build_sources(reranked_chunks)

    return {
        "answer":          answer,
        "sources":         sources,
        "rewritten_query": rewritten_query
    }


async def stream_answer(
    question: str,
    chat_history: list[dict] = [],
    doc_id: str = None
):
    """Streaming version of answer_question."""

    # Step 1: Rewrite query
    rewritten_query = rewrite_query(question, chat_history)

    # Step 2: Retrieve
    chunks = retrieve(query=rewritten_query, doc_id=doc_id)

    if not chunks:
        yield f"data: {json.dumps({'type': 'error', 'content': 'No documents found'})}\n\n"
        return

    # Step 3: Rerank
    reranked_chunks = rerank(query=rewritten_query, chunks=chunks)

    # Step 4: Build context
    context = build_context(reranked_chunks)
    sources = build_sources(reranked_chunks)

    # Step 5: Send metadata first
    yield f"data: {json.dumps({'type': 'metadata', 'sources': sources, 'rewritten_query': rewritten_query})}\n\n"

    # Step 6: Build history
    history_text = ""
    for msg in chat_history[-6:]:
        history_text += f"{msg.get('role','')}: {msg.get('content','')}\n"

    system_prompt = """You are a helpful AI assistant that answers questions
based on the provided document context.
- Answer ONLY based on the context provided
- If not found, say "I could not find this information in the uploaded documents"
- Always cite which Source you used
- Be concise and clear"""

    user_prompt = f"""Context:
{context}

Chat History:
{history_text}

Question: {question}

Answer:"""

    # Step 7: Stream from Groq
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=1024,
        stream=True
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"