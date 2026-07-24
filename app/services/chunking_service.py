import re
import numpy as np
from typing import List, Tuple
from app.services.embedding_service import embed_batch
from app.models.schemas import TranscriptSegment


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using regex.
    Handles common abbreviations and edge cases.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_chunk(
    text: str,
    threshold: float = 0.75,
    min_chunk_sentences: int = 2,
    max_chunk_sentences: int = 8,
    overlap_sentences: int = 1,  # #2: overlap between chunks
) -> List[dict]:
    """
    Splits text into semantically coherent chunks with:
    - Topic-shift detection via cosine similarity
    - Sentence overlap between adjacent chunks
    - Character offset metadata per chunk (#4)

    Returns list of dicts with 'text' and 'char_start' keys.
    """
    sentences = split_into_sentences(text)

    if len(sentences) <= min_chunk_sentences:
        return [{"text": text, "char_start": 0}] if text.strip() else []

    # Embed all sentences in one batch
    embeddings = embed_batch(sentences, batch_size=8)

    # Compute similarity between consecutive sentence pairs
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Find chunk boundaries
    boundaries = [0]  # always start a chunk at sentence 0
    for i, sim in enumerate(similarities):
        is_topic_shift = sim < threshold
        chunk_size = i + 1 - boundaries[-1]
        exceeds_max = chunk_size >= max_chunk_sentences
        below_min = chunk_size < min_chunk_sentences

        if (is_topic_shift or exceeds_max) and not below_min:
            boundaries.append(i + 1)

    boundaries.append(len(sentences))  # end boundary

    # Build chunks with overlap
    chunks = []
    for b in range(len(boundaries) - 1):
        start = boundaries[b]
        end = boundaries[b + 1]

        # Add overlap from previous chunk (#2)
        overlap_start = max(0, start - overlap_sentences)

        chunk_sentences = sentences[overlap_start:end]
        chunk_text = " ".join(chunk_sentences)

        # Compute char offset for metadata (#4)
        # Find where the first non-overlap sentence starts in original text
        prefix = " ".join(sentences[:start])
        char_start = len(prefix) + (1 if prefix else 0)

        chunks.append({
            "text": chunk_text,
            "char_start": char_start,
            "sentence_start": start,
            "sentence_end": end,
        })

    return [c for c in chunks if c["text"].strip()]


def chunk_transcript(
    full_text: str,
    segments: List[TranscriptSegment] = None,
) -> List[dict]:
    """
    Chunks transcript text semantically.
    If segments are provided, maps timestamps back to each chunk (#4).
    """
    chunks = semantic_chunk(
        full_text,
        threshold=0.75,
        min_chunk_sentences=2,
        max_chunk_sentences=8,
        overlap_sentences=1,
    )

    # Map Whisper segment timestamps back to chunks (#4)
    if segments:
        for chunk in chunks:
            # Find which segments overlap with this chunk's text
            chunk_text_lower = chunk["text"].lower()
            matching_times = []
            for seg in segments:
                if seg.text.strip().lower()[:20] in chunk_text_lower:
                    matching_times.append(seg.start)
            chunk["start_time"] = min(matching_times) if matching_times else None
    else:
        for chunk in chunks:
            chunk["start_time"] = None

    return chunks


def chunk_ocr_text(text: str) -> List[dict]:
    """
    OCR text per frame is usually short.
    Only chunk if long enough, otherwise treat as one chunk.
    """
    if len(text.split()) < 30:
        return [{"text": text, "char_start": 0, "start_time": None}]
    chunks = semantic_chunk(
        text,
        threshold=0.70,
        min_chunk_sentences=1,
        max_chunk_sentences=4,
        overlap_sentences=1,
    )
    for c in chunks:
        c["start_time"] = None
    return chunks