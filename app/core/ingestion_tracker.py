from app.core.qdrant_client import get_qdrant_client
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)
import hashlib
import uuid

TRACKER_COLLECTION = "ingested_videos"


def init_tracker_collection():
    """Creates a tracker collection if it doesn't exist."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if TRACKER_COLLECTION not in existing:
        client.create_collection(
            collection_name=TRACKER_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )


def is_duplicate(url: str) -> bool:
    """Returns True if this URL has already been ingested."""
    client = get_qdrant_client()
    url_hash = hashlib.md5(url.encode()).hexdigest()

    results = client.scroll(
        collection_name=TRACKER_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="url_hash",
                    match=MatchValue(value=url_hash),
                )
            ]
        ),
        limit=1,
    )
    return len(results[0]) > 0


def mark_as_ingested(url: str, video_id: str, video_title: str):
    """Records a URL as successfully ingested."""
    client = get_qdrant_client()
    url_hash = hashlib.md5(url.encode()).hexdigest()
    # Qdrant requires point IDs to be UUID or unsigned int — MD5 hex forms a valid UUID
    point_id = str(uuid.UUID(hex=url_hash))

    client.upsert(
        collection_name=TRACKER_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=[0.0],  # dummy vector — we only use payload for lookup
                payload={
                    "url_hash": url_hash,
                    "url": url,
                    "video_id": video_id,
                    "video_title": video_title,
                }
            )
        ]
    )
