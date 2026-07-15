from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from app.core.config import settings
from app.core.qdrant_client import get_qdrant_client
from app.models.schemas import DocumentChunk
from app.services.embedding_service import embed_text, embed_batch
from typing import List, Optional
import uuid


def index_chunks(chunks: List[DocumentChunk]) -> int:
    """
    Embeds and stores a list of DocumentChunks into Qdrant.
    Returns number of chunks successfully indexed.
    """
    if not chunks:
        return 0

    client: QdrantClient = get_qdrant_client()

    # Extract just the text for batch embedding
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_batch(texts)

    # Build Qdrant points
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "video_id": chunk.video_id,
                    "video_title": chunk.video_title,
                    "source_type": chunk.source_type,
                    "text": chunk.text,
                    "start_time": chunk.start_time,
                    "frame_path": chunk.frame_path,
                },
            )
        )

    # Upsert in batches of 50 to avoid large single payloads
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i: i + batch_size]
        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=batch,
        )

    return len(points)


def search_chunks(
    query: str,
    top_k: int = 5,
    video_id: Optional[str] = None,
) -> List[DocumentChunk]:
    """
    Embeds the query and retrieves the most similar chunks from Qdrant.
    Optionally filters by video_id to scope results to a specific video.
    """
    client: QdrantClient = get_qdrant_client()

    query_vector = embed_text(query)

    # Build optional filter for video scoping
    search_filter = None
    if video_id:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="video_id",
                    match=MatchValue(value=video_id),
                )
            ]
        )

    results = client.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )

    # Convert Qdrant results back to DocumentChunk objects
    chunks = []
    for hit in results:
        payload = hit.payload
        chunks.append(
            DocumentChunk(
                chunk_id=payload.get("chunk_id", ""),
                video_id=payload.get("video_id", ""),
                video_title=payload.get("video_title", ""),
                source_type=payload.get("source_type", ""),
                text=payload.get("text", ""),
                start_time=payload.get("start_time"),
                frame_path=payload.get("frame_path"),
            )
        )

    return chunks


def delete_video_chunks(video_id: str) -> None:
    """
    Deletes all chunks belonging to a specific video from Qdrant.
    Useful for re-ingesting an updated version of a video.
    """
    client: QdrantClient = get_qdrant_client()

    client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="video_id",
                    match=MatchValue(value=video_id),
                )
            ]
        ),
    )
    print(f"Deleted all chunks for video_id: {video_id}")