import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8003/api/v1"

st.set_page_config(
    page_title="Production RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Production RAG Assistant")
st.markdown("Upload documents and chat with them using advanced AI retrieval.")

# Session state
if "doc_id"       not in st.session_state: st.session_state.doc_id       = None
if "filename"     not in st.session_state: st.session_state.filename     = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        if st.button("Upload & Index", type="primary"):
            with st.spinner("Uploading and indexing into Pinecone..."):
                files    = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                response = requests.post(f"{API_BASE}/upload", files=files)

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.doc_id       = data["doc_id"]
                    st.session_state.filename     = data["filename"]
                    st.session_state.chat_history = []
                    st.success("✅ Indexed successfully!")
                    st.info(f"Pages: {data['total_pages']} | Chunks: {data['total_chunks']}")
                else:
                    st.error("Upload failed. Try again.")

    if st.session_state.doc_id:
        st.divider()
        st.success(f"📄 {st.session_state.filename}")
        st.code(st.session_state.doc_id[:8] + "...", language=None)

        if st.button("🗑️ Delete Document", type="secondary"):
            requests.delete(f"{API_BASE}/documents/{st.session_state.doc_id}")
            st.session_state.doc_id       = None
            st.session_state.filename     = None
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("**Pipeline:**")
    st.markdown("1. PDF → Text Extraction")
    st.markdown("2. Text → BAAI Embeddings")
    st.markdown("3. Embeddings → Pinecone")
    st.markdown("4. Query Rewriting")
    st.markdown("5. Retrieve Top 20")
    st.markdown("6. Rerank → Top 5")
    st.markdown("7. LLM → Answer + Citations")

# Main area
if not st.session_state.doc_id:
    st.info("👈 Upload a PDF from the sidebar to start chatting.")
    st.markdown("""
    ### How it works
    - Upload any PDF document
    - Ask questions in natural language
    - Get answers with source citations
    - Powered by Pinecone + Groq + Reranking + Query Rewriting
    """)
else:
    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**{src['filename']}** — "
                            f"Chunk {src['chunk_index']} — "
                            f"Score: {src['score']}"
                        )

    # Chat input
    question = st.chat_input("Ask a question about your document...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        st.session_state.chat_history.append({
            "role": "user", "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("Rewriting query → Retrieving → Reranking → Generating..."):
                response = requests.post(
                    f"{API_BASE}/chat",
                    json={
                        "question":     question,
                        "chat_history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_history
                        ],
                        "doc_id": st.session_state.doc_id
                    },
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    data            = response.json()
                    answer          = data["answer"]
                    sources         = data["sources"]
                    rewritten_query = data["rewritten_query"]

                    # Show rewritten query
                    st.caption(f"🔄 Query rewritten to: *{rewritten_query}*")
                    st.write(answer)

                    with st.expander("📚 Sources"):
                        for src in sources:
                            st.markdown(
                                f"**{src['filename']}** — "
                                f"Chunk {src['chunk_index']} — "
                                f"Score: {src['score']}"
                            )

                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error("Error getting answer. Try again.")