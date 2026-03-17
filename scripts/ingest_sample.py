#!/usr/bin/env python3
"""
Ingest Sample Documents Script
-------------------------------
Seeds the Endee vector database with the sample documents in data/sample_docs/.
Run this after starting the Endee server and the backend API.

Usage:
    python scripts/ingest_sample.py

Or, if backend is not running, set DIRECT_MODE=1 to call Endee directly.
"""
import os
import sys
import time
import pathlib
import requests

# Add parent to path for direct imports if needed
ROOT = pathlib.Path(__file__).parent.parent
SAMPLE_DIR = ROOT / "data" / "sample_docs"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def wait_for_backend(max_wait=30):
    """Wait until the backend server is accepting connections."""
    print(f"Waiting for backend at {BACKEND_URL}...")
    for i in range(max_wait):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"✅ Backend is up! ({r.json()})")
                return True
        except Exception:
            pass
        time.sleep(1)
        print(f"  [{i+1}/{max_wait}] Retrying...")
    return False


def ingest_file(filepath: pathlib.Path) -> dict:
    """Upload a file to the RAG backend for indexing."""
    print(f"\n📄 Ingesting: {filepath.name}")
    with open(filepath, "rb") as f:
        response = requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": (filepath.name, f, "text/plain")},
            timeout=120,
        )
    response.raise_for_status()
    result = response.json()
    print(f"   ✅ {result['chunks_indexed']} chunks indexed (doc_id: {result['doc_id']})")
    return result


def run_test_query():
    """Run a quick search to verify the ingestion worked."""
    print("\n🔍 Testing semantic search...")
    questions = [
        "What is machine learning?",
        "How does Endee store vectors?",
        "What are the applications of AI in healthcare?",
    ]
    for q in questions:
        r = requests.post(
            f"{BACKEND_URL}/search",
            json={"query": q, "top_k": 2},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        print(f"\n   Q: {q}")
        for i, res in enumerate(data["results"], 1):
            score = res.get("score", 0)
            preview = res.get("text", "")[:100]
            print(f"   [{i}] score={score:.3f} | {preview}...")


def main():
    if not SAMPLE_DIR.exists():
        print(f"❌ Sample docs directory not found: {SAMPLE_DIR}")
        sys.exit(1)

    files = list(SAMPLE_DIR.glob("*.txt")) + list(SAMPLE_DIR.glob("*.pdf"))
    if not files:
        print(f"❌ No .txt or .pdf files found in {SAMPLE_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("  Endee RAG — Sample Document Ingestion Script")
    print("=" * 60)

    if not wait_for_backend():
        print("❌ Backend did not start in time. Is it running?")
        print("   Run: cd backend && uvicorn app:app --reload --port 8000")
        sys.exit(1)

    results = []
    for f in files:
        try:
            r = ingest_file(f)
            results.append(r)
        except Exception as e:
            print(f"   ⚠️  Failed to ingest {f.name}: {e}")

    print(f"\n✅ Ingested {len(results)}/{len(files)} documents.")
    total_chunks = sum(r.get("chunks_indexed", 0) for r in results)
    print(f"   Total chunks in Endee: {total_chunks}")

    run_test_query()

    print("\n🎉 Done! The RAG system is ready.")
    print(f"   Open http://localhost:8000 in your browser to use the Q&A interface.")


if __name__ == "__main__":
    main()
