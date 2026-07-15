from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Video Ingestion
# ---------------------------------------------------------------------------

class VideoMetadata(BaseModel):
    """Metadata returned by yt-dlp after downloading a video."""
    video_id: str
    title: str
    uploader: Optional[str] = None
    duration: Optional[float] = None
    url: str


class TranscriptSegment(BaseModel):
    """A single timed segment from faster-whisper transcription."""
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    """Full transcription output for a video."""
    language: str
    duration: float
    full_text: str
    segments: List[TranscriptSegment]


class OCRResult(BaseModel):
    """Text extracted from a single video frame via Tesseract."""
    frame: str
    text: str


# ---------------------------------------------------------------------------
# Chunked Documents for Indexing into Qdrant
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """
    A single indexable unit stored in Qdrant.
    Each chunk has the text content, its source type,
    and metadata linking it back to the original video.
    """
    chunk_id: str
    video_id: str
    video_title: str
    source_type: str = Field(
        ...,
        description="Either 'transcript' or 'ocr'"
    )
    text: str
    start_time: Optional[float] = None  # only set for transcript chunks
    frame_path: Optional[str] = None    # only set for OCR chunks


# ---------------------------------------------------------------------------
# Compliance Audit
# ---------------------------------------------------------------------------

class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"


class ComplianceResult(BaseModel):
    """Output from the compliance auditor for a single video."""
    video_id: str
    video_title: str
    status: ComplianceStatus
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    summary: str


# ---------------------------------------------------------------------------
# API Request / Response
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """POST /ingest — request body."""
    url: str = Field(..., description="YouTube video URL to ingest and audit")


class IngestResponse(BaseModel):
    """POST /ingest — response body."""
    video_id: str
    video_title: str
    chunks_indexed: int
    compliance: ComplianceResult
    message: str


class QueryRequest(BaseModel):
    """POST /query — request body."""
    question: str = Field(..., description="Natural language compliance question")
    video_id: Optional[str] = Field(
        None,
        description="Optionally filter results to a specific video"
    )


class QueryResponse(BaseModel):
    """POST /query — response body."""
    question: str
    answer: str
    sources: List[DocumentChunk]