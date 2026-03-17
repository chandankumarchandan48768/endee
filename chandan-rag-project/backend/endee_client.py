"""
Endee Client — manages connections, index creation, upsert, and search
using the Endee vector database Python SDK.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from endee import Endee, Precision
from endee.schema import VectorItem

# Monkey-patch SDK bug where it calls .get() on a Pydantic model
if not hasattr(VectorItem, "get"):
    VectorItem.get = lambda self, key, default=None: getattr(self, key, default)

load_dotenv()

logger = logging.getLogger(__name__)

ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "documents")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


class EndeeClient:
    """Wrapper around the Endee Python SDK for RAG document indexing."""

    def __init__(self):
        self.client = Endee()
        self.client.set_base_url(f"{ENDEE_URL}/api/v1")
        self.index_name = INDEX_NAME
        self._index = None
        self._ensure_index()

    def _ensure_index(self):
        """Create the document index if it doesn't exist yet."""
        try:
            existing = self.client.list_indexes()
            if isinstance(existing, dict) and "indexes" in existing:
                index_list = existing["indexes"]
            else:
                index_list = existing or []

            names = [(idx.get("name") if isinstance(idx, dict) else getattr(idx, "name", idx)) for idx in index_list]
            if self.index_name not in names:
                logger.info(f"Creating Endee index '{self.index_name}' (dim={EMBEDDING_DIM})")
                self.client.create_index(
                    name=self.index_name,
                    dimension=EMBEDDING_DIM,
                    space_type="cosine",
                    precision=Precision.INT8,
                )
                logger.info(f"Index '{self.index_name}' created successfully.")
            else:
                logger.info(f"Index '{self.index_name}' already exists.")
            self._index = self.client.get_index(name=self.index_name)
        except Exception as e:
            logger.error(f"Error ensuring Endee index: {e}")
            raise

    @property
    def index(self):
        if self._index is None:
            self._index = self.client.get_index(name=self.index_name)
        return self._index

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Upsert document chunks into Endee.
        Each chunk: {"id": str, "vector": list[float], "meta": dict}
        """
        items = [
            {
                "id": chunk["id"],
                "vector": chunk["vector"],
                "meta": chunk["meta"],
            }
            for chunk in chunks
        ]
        self.index.upsert(items)
        logger.info(f"Upserted {len(items)} chunks into Endee index '{self.index_name}'.")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_payload: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic vector search in Endee.
        Returns list of {id, score, meta, text}.
        """
        kwargs = {"top_k": top_k}
        if filter_payload:
            kwargs["filter"] = filter_payload

        results = self.index.query(vector=query_vector, **kwargs)
        output = []
        for r in results:
            output.append(
                {
                    "id": r.get("id"),
                    "score": r.get("score", r.get("similarity", 0.0)),
                    "text": r.get("meta", {}).get("text", ""),
                    "source": r.get("meta", {}).get("source", "unknown"),
                    "chunk_index": r.get("meta", {}).get("chunk_index", 0),
                    "doc_id": r.get("meta", {}).get("doc_id", ""),
                }
            )
        return output

    def delete_by_doc_id(self, doc_id: str):
        """Delete all chunks for a given document."""
        # Endee supports filtering; we delete by iterating IDs with filter
        # Since SDK delete by filter may vary, we search broadly and delete
        try:
            dummy_vec = [0.0] * EMBEDDING_DIM
            results = self.index.query(
                vector=dummy_vec,
                top_k=512,
                filter={"doc_id": doc_id},
            )
            ids = [r.get("id") for r in results]
            if ids:
                self.index.delete(ids)
                logger.info(f"Deleted {len(ids)} chunks for doc_id='{doc_id}'.")
            return len(ids)
        except Exception as e:
            logger.warning(f"Could not delete chunks for doc_id='{doc_id}': {e}")
            return 0

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return a summary of unique ingested documents."""
        try:
            dummy_vec = [0.0] * EMBEDDING_DIM
            results = self.index.query(vector=dummy_vec, top_k=512)
            docs: Dict[str, Dict] = {}
            for r in results:
                meta = r.get("meta", {})
                doc_id = meta.get("doc_id", "")
                if doc_id and doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "source": meta.get("source", "unknown"),
                        "total_chunks": 0,
                    }
                if doc_id:
                    docs[doc_id]["total_chunks"] += 1
            return list(docs.values())
        except Exception as e:
            logger.warning(f"Could not list documents: {e}")
            return []
