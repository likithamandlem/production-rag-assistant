# rag/reranker.py
from sentence_transformers import CrossEncoder
from app.config import RERANK_TOP_N

# Load cross-encoder once at startup
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        # Free local cross-encoder — much better than score-based ranking
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def rerank(query: str, chunks: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Rerank retrieved chunks using a cross-encoder.
    
    Why reranking?
    - Pinecone retrieves top 20 by vector similarity
    - Cross-encoder scores query+chunk pairs more accurately
    - Returns only the best top_n chunks to the LLM
    - Reduces noise and improves answer quality
    """
    if not chunks:
        return []

    reranker = get_reranker()

    # Create query-chunk pairs
    pairs = [[query, chunk["text"]] for chunk in chunks]

    # Score each pair
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_n]