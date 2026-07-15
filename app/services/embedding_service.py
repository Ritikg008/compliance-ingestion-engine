from sentence_transformers import SentenceTransformer
from app.core.config import settings
from typing import List
import threading

_model = None
_model_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:

    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  
                print(f"Loading embedding model: {settings.embedding_model} ...")
                _model = SentenceTransformer(settings.embedding_model, device="cpu")
                print("Embedding model loaded.")
    return _model


def embed_text(text: str) -> List[float]:
    """Embed a single piece of text."""
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: List[str], batch_size: int = 8) -> List[List[float]]:

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.tolist()