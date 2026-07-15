from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.workflows.rag_graph import ingestion_graph
from app.services.retrieval_service import search_chunks
from app.services.compliance_auditor import audit_content
from app.core.config import settings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

router = APIRouter()


@router.get("/health")
def health_check():
    """Quick liveness check — confirms API is up."""
    return {"status": "ok", "qdrant_url": settings.qdrant_url}


@router.post("/ingest", response_model=IngestResponse)
def ingest_video(request: IngestRequest):
    """
    Full ingestion pipeline:
    Download → Transcribe → OCR → Chunk → Index → Audit
    """
    initial_state = {
        "url": request.url,
        "video_path": None,
        "metadata": None,
        "transcription": None,
        "ocr_results": None,
        "chunks": None,
        "chunks_indexed": None,
        "compliance": None,
        "error": None,
    }

    result = ingestion_graph.invoke(initial_state)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return IngestResponse(
        video_id=result["metadata"].video_id,
        video_title=result["metadata"].title,
        chunks_indexed=result["chunks_indexed"],
        compliance=result["compliance"],
        message="Video ingested and audited successfully.",
    )


@router.post("/query", response_model=QueryResponse)
def query_videos(request: QueryRequest):
    """
    RAG query endpoint:
    Embeds question → retrieves relevant chunks → generates grounded answer.
    """
    chunks = search_chunks(
        query=request.question,
        top_k=5,
        video_id=request.video_id,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found. Try ingesting a video first.",
        )

    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[{c.source_type.upper()} | {c.video_title}] {c.text}"
        for c in chunks
    ])

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
        max_tokens=1024,
    )

    messages = [
        SystemMessage(content=(
            "You are a compliance analyst assistant. "
            "Answer questions using only the provided context. "
            "If the context doesn't contain enough information, say so clearly."
        )),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Question: {request.question}"
        )),
    ]

    response = llm.invoke(messages)

    return QueryResponse(
        question=request.question,
        answer=response.content.strip(),
        sources=chunks,
    )