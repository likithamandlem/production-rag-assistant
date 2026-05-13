# rag/chain.py
from groq import Groq
from rag.retriever import retrieve
from rag.reranker import rerank
from app.config import GROQ_API_KEY, LLM_MODEL, TEMPERATURE

client = Groq(api_key=GROQ_API_KEY)

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
    """
    Full RAG pipeline:
    1. Retrieve top 20 chunks from Pinecone
    2. Rerank to top 5 using cross-encoder
    3. Build prompt with context + history
    4. Call Groq LLM
    5. Return answer + sources
    """

    # Step 1: Retrieve
    chunks = retrieve(query=question, doc_id=doc_id)

    if not chunks:
        return {
            "answer":  "No relevant documents found. Please upload documents first.",
            "sources": []
        }

    # Step 2: Rerank
    reranked_chunks = rerank(query=question, chunks=chunks)

    # Step 3: Build context
    context = build_context(reranked_chunks)

    # Step 4: Build chat history string
    history_text = ""
    for msg in chat_history[-6:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        history_text += f"{role}: {content}\n"

    # Step 5: Build prompt
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

Question: {question}

Answer:"""

    # Step 6: Call Groq
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
        "answer":  answer,
        "sources": sources
    }