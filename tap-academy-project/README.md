# Endee RAG — AI Document Q&A System

<p align="center">
  <img src="https://img.shields.io/badge/Vector%20DB-Endee-6366f1?style=for-the-badge&logo=database" />
  <img src="https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-22c55e?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LLM-Llama%203%20(Groq)-f59e0b?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi" />
</p>

> A **Retrieval Augmented Generation (RAG)** system that uses [**Endee**](https://github.com/endee-io/endee) as the vector database for semantic document search and AI-powered Q&A.

---

## 📌 Project Overview

This project demonstrates a production-ready **RAG pipeline** built with Endee as the core vector storage and retrieval layer. Users can:

1. **Upload documents** (PDF or TXT) — they are chunked, embedded, and stored in Endee.
2. **Ask natural language questions** — the system retrieves the most relevant chunks from Endee via cosine similarity search, then uses a large language model (Llama 3 via Groq) to generate a grounded, cited answer.
3. **Run semantic search** — directly query Endee to find relevant document sections purely by vector similarity.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
│          Chat Q&A │ Semantic Search │ Document Manager          │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (REST)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               FastAPI Backend (Python)                          │
│                                                                 │
│  POST /upload   ─── document_processor.py ─── PDF/TXT chunker  │
│  POST /search   ─┐                                             │
│  POST /ask      ─┤── embedder.py ─── MiniLM-L6-v2 (384 dims)  │
│                  │                                             │
│                  ▼                                             │
│            endee_client.py                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ Endee Python SDK (HTTP)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Endee Vector Database  (port 8080)                 │
│                                                                 │
│   Index: "documents"  │  Dimension: 384  │  Space: cosine      │
│   Precision: INT8     │  Payload filtering enabled             │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ (RAG path only)
                             ▼
                   ┌─────────────────┐
                   │  Groq API       │
                   │  Llama 3 8B     │
                   └─────────────────┘
```

### RAG Pipeline Flow

```
User Question
    │
    ▼
[Embed Question]  ←── sentence-transformers/all-MiniLM-L6-v2
    │
    ▼
[Search Endee]  ←── cosine similarity, top-k chunks retrieved
    │
    ▼
[Build Prompt]  ←── Question + retrieved context chunks
    │
    ▼
[LLM Generation]  ←── Groq API (Llama 3 8B)
    │
    ▼
[Return Answer + Sources]
```

---

## ⚡ How Endee Is Used

| Feature | Implementation |
|---------|---------------|
| **Index Creation** | `client.create_index(name="documents", dimension=384, space_type="cosine", precision=Precision.INT8)` |
| **Vector Upsert** | `index.upsert([{"id": chunk_id, "vector": embedding, "meta": metadata}])` |
| **Semantic Search** | `index.query(vector=query_embedding, top_k=5)` |
| **Metadata Storage** | Each chunk's text, source filename, doc_id, and chunk_index stored as payload |
| **Filtered Search** | Filter by `doc_id` to search within a specific document |
| **Bulk Delete** | Query + delete by `doc_id` for document management |

**Why Endee?**
- High-performance vector search with INT8 quantization (memory efficient)
- Simple HTTP API and clean Python SDK
- Self-hosted (no API key needed for the vector DB itself)
- Supports payload metadata for rich filtering

---

## 🚀 Setup & Installation

### Prerequisites
- **Docker** (to run Endee)
- **Python 3.10+**
- **Groq API Key** (free) — [console.groq.com](https://console.groq.com) *(optional — works without it)*

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/endee-rag-qa.git
cd endee-rag-qa
```

### Step 2: Start Endee Vector Database

```bash
docker-compose up -d
```

Endee will start on `http://localhost:8080`. Verify it's running:

```bash
curl http://localhost:8080/health
```

### Step 3: Install Python Dependencies

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (optional but recommended)
```

### Step 5: Start the Backend API

```bash
cd backend
uvicorn app:app --reload --port 8000
```

### Step 6: Open the Web Interface

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Step 7: Load Sample Documents (Optional)

```bash
# In a new terminal
python scripts/ingest_sample.py
```

This ingests pre-built documents about AI and vector databases so you can immediately ask questions without uploading your own files.

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server & Endee health check |
| `POST` | `/upload` | Upload PDF/TXT → chunk → embed → store in Endee |
| `POST` | `/search` | Semantic vector search in Endee |
| `POST` | `/ask` | Full RAG: retrieve from Endee + LLM answer |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document from Endee |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Example: Semantic Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How does machine learning work?", "top_k": 3}'
```

### Example: RAG Q&A

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the applications of AI in healthcare?", "top_k": 5}'
```

---

## 📁 Project Structure

```
.
├── docker-compose.yml          # Endee vector DB server
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── README.md
├── backend/
│   ├── app.py                  # FastAPI application
│   ├── endee_client.py         # Endee SDK wrapper
│   ├── embedder.py             # Sentence-transformer embeddings
│   ├── document_processor.py   # PDF/TXT chunking pipeline
│   └── llm_client.py           # Groq LLM integration
├── frontend/
│   ├── index.html              # Single-page web application
│   ├── style.css               # Dark premium UI styles
│   └── app.js                  # Frontend logic
├── data/
│   └── sample_docs/            # Sample documents for demo
└── scripts/
    └── ingest_sample.py        # Sample data ingestion script
```

---

## 🎯 Use Cases Demonstrated

- ✅ **RAG (Retrieval Augmented Generation)** — main Q&A feature
- ✅ **Semantic Search** — pure vector similarity search tab
- ✅ **Document Management** — upload, list, delete documents in Endee
- ✅ **Hybrid Metadata Filtering** — filter search by document ID
- ✅ **Payload Storage** — text chunks stored as Endee metadata

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector Database | **Endee** (self-hosted, Docker) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (384 dims) |
| LLM | `Llama 3 8B` via Groq API |
| Backend | **FastAPI** + Python |
| Frontend | Vanilla HTML / CSS / JavaScript |
| PDF Parsing | PyMuPDF |

---

## 📝 License

MIT

---

*Built for the Endee.io × Tap Academy assignment — demonstrating a practical RAG application powered by the Endee vector database.*
