# rag/ingestion.py
import uuid
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from rag.embeddings import embed_texts
from app.config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP
)

# Initialize Pinecone
pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def extract_text(file_path: str) -> tuple[str, int]:
    """Extract text from PDF."""
    reader = PdfReader(file_path)
    text   = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text, len(reader.pages)

def chunk_text(text: str) -> list[str]:
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

def ingest_document(file_path: str, filename: str) -> dict:
    """Full ingestion pipeline: PDF → chunks → embeddings → Pinecone."""

    # Step 1: Extract text
    text, total_pages = extract_text(file_path)
    if not text.strip():
        raise ValueError("Could not extract text from PDF")

    # Step 2: Chunk
    chunks = chunk_text(text)

    # Step 3: Embed
    embeddings = embed_texts(chunks)

    # Step 4: Generate document ID
    doc_id = str(uuid.uuid4())

    # Step 5: Upsert to Pinecone with metadata
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id":     f"{doc_id}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "doc_id":      doc_id,
                "filename":    filename,
                "chunk_index": i,
                "text":        chunk[:1000],
            }
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

    return {
        "doc_id":       doc_id,
        "filename":     filename,
        "total_pages":  total_pages,
        "total_chunks": len(chunks)
    }

def delete_document(doc_id: str):
    """Delete all chunks of a document from Pinecone."""
    index.delete(filter={"doc_id": {"$eq": doc_id}})

def list_documents() -> list[dict]:
    """List all documents stored in Pinecone."""
    stats = index.describe_index_stats()
    return {"total_vectors": stats.total_vector_count}