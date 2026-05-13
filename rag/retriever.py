# rag/retriever.py
from pinecone import Pinecone
from rag.embeddings import embed_query
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, TOP_K

# Initialize Pinecone
pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def retrieve(query: str, doc_id: str = None, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve top_k relevant chunks from Pinecone.
    If doc_id is provided, search only within that document.
    """

    # Embed the query
    query_embedding = embed_query(query)

    # Build filter
    filter_dict = None
    if doc_id:
        filter_dict = {"doc_id": {"$eq": doc_id}}

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )

    # Format results
    chunks = []
    for match in results.matches:
        chunks.append({
            "text":        match.metadata.get("text", ""),
            "filename":    match.metadata.get("filename", ""),
            "doc_id":      match.metadata.get("doc_id", ""),
            "chunk_index": match.metadata.get("chunk_index", 0),
            "score":       match.score
        })

    return chunks