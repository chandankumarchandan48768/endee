"""
LLM Client — uses Groq API to generate answers from retrieved context.
Falls back to a context-only response if no API key is configured.
"""
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided document context.

Instructions:
- Answer the user's question using ONLY the information from the provided context.
- Be concise, accurate, and helpful.
- If the context doesn't contain enough information to answer, say so clearly.
- Cite source documents when relevant.
- Format your response in a clear, readable way."""


def _build_context_str(chunks: List[Dict[str, Any]]) -> str:
    """Build a formatted context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        score = chunk.get("score", 0.0)
        parts.append(
            f"[Source {i}: {source} | Relevance: {score:.2f}]\n{text}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate an LLM-powered answer using retrieved context from Endee.

    Returns:
        {
            "answer": str,
            "model": str,
            "used_llm": bool,
            "chunks_used": int
        }
    """
    context = _build_context_str(retrieved_chunks)

    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        # Graceful fallback: return top retrieved chunks as the "answer"
        logger.warning("No GROQ_API_KEY set — returning retrieved context only.")
        answer = (
            "⚠️ No LLM API key configured. Here are the most relevant document sections:\n\n"
            + context
        )
        return {
            "answer": answer,
            "model": "none (no API key)",
            "used_llm": False,
            "chunks_used": len(retrieved_chunks),
        }

    try:
        from groq import Groq

        groq_client = Groq(api_key=GROQ_API_KEY)

        user_message = f"""Context from documents:
{context}

Question: {question}

Please answer the question based on the context above."""

        completion = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        answer = completion.choices[0].message.content.strip()
        logger.info(f"LLM answered using model '{LLM_MODEL}'.")
        return {
            "answer": answer,
            "model": LLM_MODEL,
            "used_llm": True,
            "chunks_used": len(retrieved_chunks),
        }

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        fallback = (
            f"⚠️ LLM error ({e}). Here are the most relevant document sections:\n\n"
            + context
        )
        return {
            "answer": fallback,
            "model": "error",
            "used_llm": False,
            "chunks_used": len(retrieved_chunks),
        }
