# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# API Keys
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Pinecone
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-assistant")

# LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]     = LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"]     = "production-rag-assistant"

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384

# Chunking
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K        = 20   # fetch top 20 from Pinecone
RERANK_TOP_N = 5    # rerank to top 5 for LLM

# LLM
LLM_MODEL   = "llama-3.3-70b-versatile"
TEMPERATURE = 0

# Directories
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)