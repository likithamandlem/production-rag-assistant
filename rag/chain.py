# rag/chain.py
from groq import Groq
from rag.retriever import retrieve
from rag.reranker import rerank
from app.config import GROQ_API_KEY, LLM_MODEL, TEMPERATURE

client = Groq(api_key=GROQ_API_KEY)

def rewrite_query(question: str, chat_history: list[dict] = []) -> str:
    """
    Rewrite user query to improve retrieval quality.
    
    Example:
    User asks: "termination?"
    Rewritten: "What are the termination conditions, notice periods,
                penalties and cancellation clauses in this document?"
    
    This improves retrieval because:
    - Short queries miss relevant chunks
    - Rewritten queries match more document vocabulary
    - Context from chat history improves follow-up questions
    """
    history_text = ""
    for msg in chat_history[-4:]:
        history_text += f"{msg.get('role','')}: {msg.get('content','')}\n"

    prompt = f"""You are a query rewriting expert for a RAG system.
Your job is to rewrite the user's question to improve document retrieval.

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
    rewritten = response.choices[0].message.content.strip()
    return rewritten

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
    Full RAG pipeline with query rewriting:
    1. Rewrite query for better retrieval
    2. Retrieve top 20 chunks from Pinecone
    3. Rerank to top 5 using cross-encoder
    4. Build prompt with context + history
    5. Call Groq LLM
    6. Return answer + sources + rewritten query
    """

    # Step 1: Rewrite query
    rewritten_query = rewrite_query(question, chat_history)

    # Step 2: Retrieve using REWRITTEN query
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

    # Step 5: Build chat history string
    history_text = ""
    for msg in chat_history[-6:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        history_text += f"{role}: {content}\n"

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