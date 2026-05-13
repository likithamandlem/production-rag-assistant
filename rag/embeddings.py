# rag/embeddings.py
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL

# Load once at startup
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert list of texts to list of embedding vectors."""
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  # important for cosine similarity
        show_progress_bar=False
    )
    return embeddings.tolist()

def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = get_embedding_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True
    )
    return embedding.tolist()