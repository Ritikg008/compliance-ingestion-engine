from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from pathlib import Path
from app.services.chunking_service import chunk_transcript, chunk_ocr_text

from app.models.schemas import (
    VideoMetadata,
    TranscriptionResult,
    OCRResult,
    DocumentChunk,
    ComplianceResult,
)
from app.services.video_processor import download_video, extract_frames, cleanup_temp_files
from app.services.transcription_service import transcribe_video, unload_whisper_model
from app.services.ocr_service import extract_text_from_frames
from app.services.retrieval_service import index_chunks, search_chunks
from app.services.compliance_auditor import audit_content
import uuid



# State — the shared object passed between every node in the graph


class IngestionState(TypedDict):
    # Input
    url: str

    # Set by download_node
    video_path: Optional[Path]
    metadata: Optional[VideoMetadata]

    # Set by transcription_node
    transcription: Optional[TranscriptionResult]

    # Set by ocr_node
    ocr_results: Optional[List[OCRResult]]

    # Set by chunking_node
    chunks: Optional[List[DocumentChunk]]

    # Set by indexing_node
    chunks_indexed: Optional[int]

    # Set by compliance_node
    compliance: Optional[ComplianceResult]

    # Error handling
    error: Optional[str]


# Nodes — each is one step in the pipeline


def download_node(state: IngestionState) -> IngestionState:
    """Downloads the video and extracts metadata."""
    print(" [Node 1/6] Downloading video...")
    try:
        path, meta = download_video(state["url"])
        return {
            **state,
            "video_path": path,
            "metadata": VideoMetadata(
                video_id=meta["video_id"],
                title=meta["title"],
                uploader=meta.get("uploader"),
                duration=meta.get("duration"),
                url=state["url"],
            ),
            "error": None,
        }
    except Exception as e:
        return {**state, "error": f"Download failed: {str(e)}"}


def transcription_node(state: IngestionState) -> IngestionState:
    """Transcribes audio from the downloaded video."""
    if state.get("error"):
        return state
    print(" [Node 2/6] Transcribing audio...")
    try:
        result = transcribe_video(state["video_path"])
        unload_whisper_model() 
        from app.models.schemas import TranscriptSegment, TranscriptionResult as TR
        return {
            **state,
            "transcription": TR(
                language=result["language"],
                duration=result["duration"],
                full_text=result["full_text"],
                segments=[
                    TranscriptSegment(**seg) for seg in result["segments"]
                ],
            ),
        }
    except Exception as e:
        return {**state, "error": f"Transcription failed: {str(e)}"}


def ocr_node(state: IngestionState) -> IngestionState:
    """Extracts frames and runs OCR on them."""
    if state.get("error"):
        return state
    print(" [Node 3/6] Extracting frames and running OCR...")
    try:
        frame_paths = extract_frames(state["video_path"], interval_seconds=5)
        raw_results = extract_text_from_frames(frame_paths)
        # Delete frames only — video cleanup happens in indexing_node
        for fp in frame_paths:
            if fp.exists():
                fp.unlink()
        return {
            **state,
            "ocr_results": [
                OCRResult(frame=r["frame"], text=r["text"])
                for r in raw_results
            ],
        }
    except Exception as e:
        return {**state, "error": f"OCR failed: {str(e)}"}

def chunking_node(state: IngestionState) -> IngestionState:
    """
    Converts transcription and OCR into semantically coherent chunks
    with overlap and timestamp metadata.
    """
    if state.get("error"):
        return state
    print("[Node 4/6] Semantic chunking with overlap...")
    try:
        chunks = []
        meta = state["metadata"]

        # Semantic chunking of full transcript with timestamp mapping
        if state.get("transcription"):
            full_text = state["transcription"].full_text
            segments = state["transcription"].segments
            if full_text.strip():
                transcript_chunks = chunk_transcript(full_text, segments)
                print(f"    Transcript → {len(transcript_chunks)} semantic chunks")
                for chunk_data in transcript_chunks:
                    if chunk_data["text"].strip():
                        chunks.append(DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            video_id=meta.video_id,
                            video_title=meta.title,
                            source_type="transcript",
                            text=chunk_data["text"].strip(),
                            start_time=chunk_data.get("start_time"),
                        ))

        # Semantic chunking of OCR text per frame
        if state.get("ocr_results"):
            for ocr in state["ocr_results"]:
                if ocr.text.strip():
                    ocr_chunks = chunk_ocr_text(ocr.text)
                    for chunk_data in ocr_chunks:
                        if chunk_data["text"].strip():
                            chunks.append(DocumentChunk(
                                chunk_id=str(uuid.uuid4()),
                                video_id=meta.video_id,
                                video_title=meta.title,
                                source_type="ocr",
                                text=chunk_data["text"].strip(),
                                frame_path=ocr.frame,
                            ))

        print(f"    Total chunks: {len(chunks)}")
        return {**state, "chunks": chunks}
    except Exception as e:
        return {**state, "error": f"Chunking failed: {str(e)}"}

    
def indexing_node(state: IngestionState) -> IngestionState:
    """Embeds and stores all chunks into Qdrant."""
    if state.get("error"):
        return state
    print(" [Node 5/6] Indexing chunks into Qdrant...")
    try:
        count = index_chunks(state["chunks"])
        if state["video_path"].exists():
            state["video_path"].unlink()
        return {**state, "chunks_indexed": count}
    except Exception as e:
        return {**state, "error": f"Indexing failed: {str(e)}"}


def compliance_node(state: IngestionState) -> IngestionState:
    """Retrieves relevant chunks and runs the compliance audit."""
    if state.get("error"):
        return state
    print(" [Node 6/6] Running compliance audit...")
    try:
        meta = state["metadata"]
        relevant_chunks = search_chunks(
            query="compliance issues claims guarantees disclaimers",
            top_k=10,
            video_id=meta.video_id,
        )
        result = audit_content(relevant_chunks, meta.video_id, meta.title)
        return {**state, "compliance": result}
    except Exception as e:
        return {**state, "error": f"Compliance audit failed: {str(e)}"}


# Error check — routes to END if any node set an error

def should_continue(state: IngestionState) -> str:
    if state.get("error"):
        print(f" Pipeline stopped: {state['error']}")
        return "end"
    return "continue"

# Build the graph

def build_ingestion_graph() -> StateGraph:
    graph = StateGraph(IngestionState)

    graph.add_node("node_download", download_node)
    graph.add_node("node_transcription", transcription_node)
    graph.add_node("node_ocr", ocr_node)
    graph.add_node("node_chunking", chunking_node)
    graph.add_node("node_indexing", indexing_node)
    graph.add_node("node_compliance", compliance_node)

    graph.set_entry_point("node_download")

    for src, dst in [
        ("node_download", "node_transcription"),
        ("node_transcription", "node_ocr"),
        ("node_ocr", "node_chunking"),
        ("node_chunking", "node_indexing"),
        ("node_indexing", "node_compliance"),
    ]:
        graph.add_conditional_edges(
            src,
            should_continue,
            {"continue": dst, "end": END},
        )

    graph.add_edge("node_compliance", END)

    return graph.compile()


# Compiled graph — imported by FastAPI routes
ingestion_graph = build_ingestion_graph()