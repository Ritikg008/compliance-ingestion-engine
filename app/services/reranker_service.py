from sentence_transformers import CrossEncoder
from app.models.schemas import DocumentChunk
from typing import List
import threading

_reranker = None
_reranker_lock = threading.Lock()


def get_reranker() -> CrossEncoder:
    """Lazy-loads cross-encoder once and reuses it."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                print("Loading reranker model...")
                _reranker = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    max_length=512,
                )
                print("Reranker loaded.")
    return _reranker


def rerank_chunks(
    query: str,
    chunks: List[DocumentChunk],
    top_k: int = 10,
) -> List[DocumentChunk]:
    """
    Re-scores chunks using a cross-encoder and returns top_k results.

    Unlike bi-encoder (embedding similarity), a cross-encoder sees
    both query and chunk together — much more accurate relevance scoring
    but too slow to run on all vectors, so we run it after Qdrant retrieval.
    """
    if not chunks:
        return []

    reranker = get_reranker()

    # Build query-chunk pairs for cross-encoder
    pairs = [[query, chunk.text] for chunk in chunks]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Sort by score descending and return top_k
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]