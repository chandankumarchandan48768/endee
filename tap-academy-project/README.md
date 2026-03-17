# 🧠 Endee RAG — AI Document Q&A System

> A Retrieval-Augmented Generation (RAG) system powered by the **Endee Vector Database**, enabling intelligent, context-aware answers to natural language questions over your own documents.

---

## 📌 Introduction

This project is built as part of the **Tap Academy × Endee.io** assignment. Candidates were required to design and develop an AI/ML application using [Endee](https://github.com/endee-io/endee) as the vector database, demonstrating a practical use case.

This project implements a full **RAG (Retrieval-Augmented Generation)** pipeline — upload any PDF or text document, and ask natural language questions. The system finds the most semantically relevant content using Endee's vector search, then passes it to a powerful LLM (via Groq) to generate a clear, well-structured answer — just like ChatGPT, but over *your* documents.

---

## ✨ Features

- 📄 **Document Upload** — Ingest PDF, TXT, and Markdown files
- 🔍 **Semantic Search** — Vector similarity search powered by Endee
- 🤖 **AI Q&A (RAG)** — LLM-generated answers grounded in your documents
- 💻 **Code Highlighting** — Syntax-highlighted code blocks with one-click copy
- 📊 **Source Citations** — Answers include relevance scores and source references
- 🌐 **Full-Stack Web App** — Clean, modern UI served directly from FastAPI

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User's Browser                       │
│          (HTML + CSS + JS  →  http://localhost:8000)    │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP Requests (REST API)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend  (Python)                  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ /upload      │  │  /ask (RAG)  │  │  /search     │  │
│  │ Chunk + Embed│  │ Embed + Query│  │ Embed + Query│  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
│  ┌──────▼─────────────────▼─────────────────▼───────┐  │
│  │       Sentence-Transformers Embedder              │  │
│  │       (all-MiniLM-L6-v2, dim=384)                 │  │
│  └──────────────────────┬────────────────────────────┘  │
└─────────────────────────┼───────────────────────────────┘
                          │ Vector Upsert / Query
                          ▼
┌─────────────────────────────────────────────────────────┐
│        Endee Vector Database  (Docker Container)        │
│              http://localhost:8080                      │
│   Index: "documents"  |  Space: cosine  |  dim: 384     │
└─────────────────────────────────────────────────────────┘
                          │ Retrieved Top-K Chunks
                          ▼
┌─────────────────────────────────────────────────────────┐
│             Groq LLM API  (llama-3.1-8b-instant)        │
│        Generates a well-structured Markdown answer      │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Application Flow

### 1. 📤 Document Ingestion
```
User Uploads File
      │
      ▼
Document Processor  ──→  Split into ~500-char chunks
      │
      ▼
Sentence-Transformer  ──→  Embed each chunk → 384-dim vector
      │
      ▼
Endee Vector DB  ──→  Upsert vectors with metadata (source, chunk_index, doc_id)
```

### 2. 🔍 Semantic Search
```
User types a query
      │
      ▼
Embed query  ──→  384-dim vector
      │
      ▼
Endee cosine similarity search  ──→  Top-K most relevant chunks
      │
      ▼
Return results with relevance scores
```

### 3. 🤖 AI Q&A (RAG Pipeline)
```
User asks a question
      │
      ▼
Embed question  ──→  vector
      │
      ▼
Endee retrieves Top-K relevant chunks
      │
      ▼
Build context prompt:  [System Prompt] + [Retrieved Chunks] + [User Question]
      │
      ▼
Groq LLM (llama-3.1-8b-instant)  ──→  Generates Markdown-formatted answer
      │
      ▼
Frontend renders response with syntax highlighting + copy buttons
```

---

## 🗂️ Project Structure

```
tap-academy-project/
├── backend/
│   ├── app.py               # FastAPI application & all API endpoints
│   ├── endee_client.py      # Endee vector DB wrapper (index, upsert, search)
│   ├── embedder.py          # Sentence-transformers embedding logic
│   ├── document_processor.py# PDF/TXT chunking logic
│   └── llm_client.py        # Groq LLM integration
├── frontend/
│   ├── index.html           # Main SPA with marked.js + highlight.js
│   ├── style.css            # Dark-mode premium UI styles
│   └── app.js               # Frontend logic (chat, search, upload)
├── data/
│   └── sample_docs/         # Pre-built AI & Vector DB sample texts
├── scripts/
│   └── ingest_sample.py     # CLI script to seed the database
├── docker-compose.yml        # Endee vector DB container
├── .env                     # Environment variables (API keys, config)
└── requirements.txt         # Python dependencies
```

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.10+
- Docker Desktop (running)
- A free [Groq API Key](https://console.groq.com)

### 1. Clone the Repository
```bash
git clone https://github.com/chandankumarchandan48768/endee
cd endee/tap-academy-project
```

### 2. Start Endee Vector Database
```bash
docker compose up -d
```

### 3. Set Up Python Environment
```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Edit `.env` and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.1-8b-instant
ENDEE_URL=http://localhost:8080
```

### 5. Start the Backend
```bash
cd backend
uvicorn app:app --reload --port 8000
```

### 6. Open the App
Visit **[http://localhost:8000](http://localhost:8000)** in your browser.

### 7. (Optional) Seed Sample Documents
```bash
# In a new terminal, from tap-academy-project/
source venv/bin/activate
python scripts/ingest_sample.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Vector Database | **Endee** (self-hosted via Docker) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | **Llama 3.1** via **Groq API** |
| Backend | **FastAPI** (Python) |
| Frontend | HTML + CSS + Vanilla JS |
| Markdown Rendering | `marked.js` + `highlight.js` |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server & Endee connection status |
| `POST` | `/upload` | Upload & ingest a document |
| `POST` | `/ask` | Full RAG pipeline — ask a question |
| `POST` | `/search` | Pure semantic vector search |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{id}` | Delete a document from Endee |

---

## 📖 Why Endee?

[Endee](https://github.com/endee-io/endee) is a lightweight, high-performance vector database designed for AI applications. In this project, Endee is used to:

- **Store** document embeddings with rich metadata
- **Index** vectors using cosine similarity in an HNSW graph
- **Retrieve** the most semantically relevant chunks in milliseconds

This enables the RAG system to ground LLM responses in factual, user-provided content — eliminating hallucinations and providing source-cited answers.

---

## 👤 Author

**Chandan Kumar K N**  
GitHub: [chandankumarchandan48768](https://github.com/chandankumarchandan48768)  
Email: chandankumar.springdev@gmail.com

---

*Built for Tap Academy × Endee.io Assignment — March 2026*
