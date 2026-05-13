# 🧠 Production RAG Assistant

An advanced production-grade Retrieval-Augmented Generation (RAG) system
that allows users to upload documents and chat with them using state-of-the-art
AI retrieval, reranking, and LLM generation.

---

# 🚀 Features

- 📄 **PDF Upload & Indexing** — documents stored in Pinecone cloud vector DB
- 🔍 **Semantic Retrieval** — BAAI/bge-small-en-v1.5 embeddings for accurate search
- 🎯 **Cross-encoder Reranking** — retrieves top 20, reranks to best 5
- 💬 **Conversational RAG** — chat with memory and context
- 📚 **Source Citations** — every answer cites exact source chunks
- 🗑️ **Document Management** — upload and delete documents
- 🔭 **LangSmith Tracing** — full LLM observability
- 🐳 **Dockerized** — production-ready deployment

---

# 🏗️ System Architecture

```text
PDF Upload
    ↓
Text Extraction (PyPDF)
    ↓
Chunking (1000 chars, 200 overlap)
    ↓
BAAI/bge-small-en-v1.5 Embeddings
    ↓
Pinecone Cloud Vector Store
    ↓
Query → Embed → Retrieve Top 20
    ↓
Cross-encoder Reranking → Top 5
    ↓
Groq Llama 3.3 70B → Answer + Citations
```

---

# 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM | Llama 3.3 70B via Groq |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Database | Pinecone (Cloud) |
| Reranking | Cross-encoder ms-marco-MiniLM-L-6-v2 |
| Monitoring | LangSmith |
| Containerisation | Docker |
| Language | Python 3.11 |

---

# 📂 Project Structure

```text
production-rag-assistant/
│
├── app/
│   ├── api/routes/
│   │   ├── upload.py      # PDF upload + Pinecone indexing
│   │   └── chat.py        # RAG chat endpoint
│   ├── config.py          # centralised settings
│   └── main.py            # FastAPI app
│
├── rag/
│   ├── embeddings.py      # BAAI embedding model
│   ├── ingestion.py       # PDF → chunks → Pinecone
│   ├── retriever.py       # semantic search
│   ├── reranker.py        # cross-encoder reranking
│   └── chain.py           # full RAG pipeline
│
├── frontend/
│   └── streamlit_app.py   # Streamlit UI
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# ⚙️ Setup & Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/likithamandlem/production-rag-assistant.git
cd production-rag-assistant
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Setup Environment Variables

Create `.env` file:

```env
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_HOST=your_pinecone_host
LANGSMITH_API_KEY=your_langsmith_key
PINECONE_INDEX_NAME=rag-assistant
```

---

# ▶️ Running the Application

## FastAPI Backend

```bash
uvicorn app.main:app --reload --port 8003
```

API Docs: http://127.0.0.1:8003/docs

## Streamlit Frontend

```bash
streamlit run frontend/streamlit_app.py
```

Frontend: http://localhost:8501

---

# 🐳 Docker Setup

```bash
docker build -t production-rag-assistant .
docker run -p 8003:8003 --env-file .env production-rag-assistant
```

---

# 💬 Example Interaction

**Upload:** Any PDF document

**Ask:** "What are the main topics covered?"

**Response:**
The document covers:

Topic 1 (Source 1)
Topic 2 (Source 2)
Topic 3 (Source 3)


---

# 🔄 RAG Pipeline Detail

| Step | Detail |
|---|---|
| Retrieval | Top 20 chunks from Pinecone |
| Reranking | Cross-encoder scores all 20, returns best 5 |
| Context | Best 5 chunks sent to LLM |
| Generation | Llama 3.3 70B generates grounded answer |
| Citations | Every answer references source chunks |

---

# 🧩 Challenges Faced

- Pinecone SDK v3 requires host URL separately from API key
- Cross-encoder reranking adds latency but significantly improves answer quality
- LangChain text_splitter moved to separate package in newer versions
- Balancing chunk size vs retrieval precision

---

# 🔮 Future Improvements

- Add hybrid search (dense + sparse BM25)
- Add streaming responses
- Add user authentication
- Deploy to AWS ECS
- Add evaluation pipeline with RAGAS
- Support multiple file formats (DOCX, TXT, CSV)

---

# 🎯 Target Roles

- AI Engineer
- GenAI Engineer
- LLM Engineer
- Applied AI Engineer
- ML Platform Engineer

---

# 👩‍💻 Author

**Likitha Mandlem**
GitHub: https://github.com/likithamandlem