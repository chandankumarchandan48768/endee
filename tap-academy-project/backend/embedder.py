"""
Embedder — generates sentence embeddings using sentence-transformers.
Uses all-MiniLM-L6-v2 (384 dimensions) for fast, high-quality embeddings.
"""
import os
import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Load model once at module import
_model: SentenceTransformer = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _model


def embed(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts into 384-dimensional float vectors.
    Returns list of vectors (one per text).
    """
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed_single(text: str) -> List[float]:
    """Embed a single text string."""
    return embed([text])[0]
