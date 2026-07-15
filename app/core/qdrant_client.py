from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.core.config import settings


def get_qdrant_client() -> QdrantClient:
    """Returns a singleton-style Qdrant client instance."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def init_collection():
    """Creates the Qdrant collection if it doesn't already exist."""
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    existing_collections = [c.name for c in client.get_collections().collections]

    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        print(f" Created Qdrant collection: '{collection_name}'")
    else:
        print(f" Collection '{collection_name}' already exists, skipping creation.")


if __name__ == "__main__":
    init_collection()