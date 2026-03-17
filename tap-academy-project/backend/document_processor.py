"""
Document Processor — handles PDF and TXT file parsing and text chunking.
Produces overlapping chunks suitable for RAG retrieval.
"""
import re
import uuid
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512       # characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks


def _read_txt(file_bytes: bytes) -> str:
    """Decode raw bytes as UTF-8 text."""
    return file_bytes.decode("utf-8", errors="replace")


def _read_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    except Exception as e:
        logger.error(f"PDF reading failed: {e}")
        raise ValueError(f"Could not read PDF: {e}")


def _clean_text(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks of CHUNK_SIZE characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def process_file(
    file_bytes: bytes,
    filename: str,
    doc_id: str = None,
) -> List[Dict[str, Any]]:
    """
    Parse a file and return a list of chunk dicts:
      {
        "id": unique chunk id,
        "text": chunk text,
        "meta": {
          "doc_id": str,
          "source": filename,
          "chunk_index": int,
          "text": str      # stored in Endee metadata for retrieval
        }
      }
    """
    if doc_id is None:
        doc_id = str(uuid.uuid4())

    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        raw_text = _read_pdf(file_bytes)
    elif ext in (".txt", ".md"):
        raw_text = _read_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or TXT.")

    clean = _clean_text(raw_text)
    if not clean:
        raise ValueError("Document appears to be empty after text extraction.")

    text_chunks = _chunk_text(clean)
    result = []
    for i, chunk in enumerate(text_chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        result.append(
            {
                "id": chunk_id,
                "text": chunk,
                "meta": {
                    "doc_id": doc_id,
                    "source": filename,
                    "chunk_index": i,
                    "text": chunk,  # stored in Endee metadata for retrieval
                },
            }
        )

    logger.info(
        f"Processed '{filename}': {len(clean)} chars → {len(result)} chunks."
    )
    return result
