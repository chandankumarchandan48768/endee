"""
Main FastAPI Application — RAG Document Q&A System powered by Endee Vector Database.

Endpoints:
  GET  /health        — server + Endee health check
  POST /upload        — upload & ingest a document into Endee
  POST /search        — semantic vector search over indexed documents
  POST /ask           — full RAG: search + LLM answer generation
  GET  /documents     — list all ingested documents
  DELETE /documents/{doc_id} — remove a document from Endee
"""
import os
import uuid
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# -- Lazy import Endee client to avoid crash if server isn't up yet --
from endee_client import EndeeClient
from embedder import embed_single
from document_processor import process_file
from llm_client import generate_answer

app = FastAPI(
    title="Endee RAG Document Q&A",
    description=(
        "Retrieval Augmented Generation (RAG) system using Endee as the vector database. "
        "Upload documents, then ask natural language questions answered with AI."
    ),
    version="1.0.0",
)

# CORS — allow frontend served from disk or any local port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend at /
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
# Static files will be mounted at the bottom of the file to not override the API routes


# --- Endee client (initialised once) ---
_db: Optional[EndeeClient] = None


def get_db() -> EndeeClient:
    global _db
    if _db is None:
        _db = EndeeClient()
    return _db


# ── Pydantic models ─────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_id: Optional[str] = None   # optionally filter by document


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_id: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Health check: verifies the API server and attempts to reach Endee."""
    try:
        db = get_db()
        return {
            "status": "ok",
            "endee": "connected",
            "index": db.index_name,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Endee not reachable: {e}")


@app.post("/upload", tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(default=None),
):
    """
    Upload a PDF or TXT document. The file is chunked, embedded, and
    stored in the Endee vector database for semantic retrieval.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    allowed_exts = {".pdf", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_exts}",
        )

    doc_id = doc_id or str(uuid.uuid4())
    content = await file.read()

    try:
        # 1. Parse & chunk
        chunks = process_file(content, file.filename, doc_id=doc_id)
        if not chunks:
            raise HTTPException(status_code=422, detail="No text extracted from file.")

        # 2. Embed all chunks (batch)
        texts = [c["text"] for c in chunks]
        vectors = embed_chunk_batch(texts)

        # 3. Attach vectors and upsert into Endee
        records = []
        for chunk, vec in zip(chunks, vectors):
            records.append({"id": chunk["id"], "vector": vec, "meta": chunk["meta"]})

        db = get_db()
        db.upsert_chunks(records)

        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks_indexed": len(records),
            "message": f"Successfully indexed {len(records)} chunks into Endee.",
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")


@app.post("/search", tags=["Retrieval"])
async def semantic_search(req: SearchRequest):
    """
    Pure semantic vector search using Endee.
    Returns the top-k most relevant document chunks with similarity scores.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        query_vec = embed_single(req.query)
        db = get_db()

        filter_payload = {"doc_id": req.doc_id} if req.doc_id else None
        results = db.search(query_vec, top_k=req.top_k, filter_payload=filter_payload)

        return {
            "query": req.query,
            "top_k": req.top_k,
            "results": results,
            "total_found": len(results),
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")


@app.post("/ask", tags=["RAG"])
async def ask_question(req: AskRequest):
    """
    Full RAG pipeline:
    1. Embed the question
    2. Retrieve top-k relevant chunks from Endee
    3. Pass question + context to LLM (Groq)
    4. Return the AI-generated answer with sources
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Step 1 — Embed query
        query_vec = embed_single(req.question)

        # Step 2 — Retrieve from Endee
        db = get_db()
        filter_payload = {"doc_id": req.doc_id} if req.doc_id else None
        retrieved = db.search(query_vec, top_k=req.top_k, filter_payload=filter_payload)

        if not retrieved:
            return {
                "question": req.question,
                "answer": "No relevant documents found. Please upload some documents first.",
                "sources": [],
                "model": "none",
                "used_llm": False,
            }

        # Step 3 — Generate answer with LLM
        response = generate_answer(req.question, retrieved)

        # Step 4 — Return structured result
        sources = [
            {
                "source": r["source"],
                "chunk_index": r["chunk_index"],
                "score": round(r["score"], 4),
                "preview": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
            }
            for r in retrieved
        ]

        return {
            "question": req.question,
            "answer": response["answer"],
            "sources": sources,
            "model": response["model"],
            "used_llm": response["used_llm"],
            "chunks_used": response["chunks_used"],
        }

    except Exception as e:
        logger.error(f"Ask failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")


@app.get("/documents", tags=["Documents"])
async def list_documents():
    """List all documents currently indexed in the Endee vector database."""
    try:
        db = get_db()
        docs = db.list_documents()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {e}")


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Remove all chunks of a document from the Endee vector database."""
    try:
        db = get_db()
        deleted = db.delete_by_doc_id(doc_id)
        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks_deleted": deleted,
        }
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Delete error: {e}")


# ── Helper ────────────────────────────────────────────────────────────────────

def embed_chunk_batch(texts: list) -> list:
    """Batch embed texts using the embedder module."""
    from embedder import embed
    return embed(texts)

# Catch-all to serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
