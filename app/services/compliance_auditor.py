from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import settings
from app.models.schemas import DocumentChunk, ComplianceResult, ComplianceStatus
from typing import List
import json


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,      
        max_tokens=1024,
    )


def build_audit_prompt(chunks: List[DocumentChunk]) -> str:
    content_blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{chunk.source_type.upper()}]"
        time_info = f" @ {chunk.start_time}s" if chunk.start_time else ""
        content_blocks.append(
            f"{i}. {source}{time_info}\n{chunk.text}"
        )

    content_str = "\n\n".join(content_blocks)

    return f"""You are a compliance auditor reviewing video content.
Analyze the following extracted content (from transcription and OCR) for compliance issues.

Look for:
- Unsubstantiated health or medical claims
- Misleading financial claims or guarantees
- False advertising or exaggerated product claims
- Missing required disclaimers
- Potentially illegal or deceptive statements
- Privacy or data protection concerns

CONTENT TO AUDIT:
{content_str}

Respond ONLY with a valid JSON object in this exact format:
{{
    "status": "compliant" | "non_compliant" | "needs_review",
    "issues": ["issue 1", "issue 2"],
    "recommendations": ["recommendation 1", "recommendation 2"],
    "summary": "one paragraph summary of your findings"
}}

No extra text, no markdown, just the JSON object."""


def audit_content(
    chunks: List[DocumentChunk],
    video_id: str,
    video_title: str,
) -> ComplianceResult:
    if not chunks:
        return ComplianceResult(
            video_id=video_id,
            video_title=video_title,
            status=ComplianceStatus.NEEDS_REVIEW,
            issues=["No content could be extracted from this video for analysis."],
            recommendations=["Verify the video has audible speech or visible text."],
            summary="Insufficient content was extracted to perform a compliance audit.",
        )

    llm = get_llm()
    prompt = build_audit_prompt(chunks)

    messages = [
        SystemMessage(content=(
            "You are a strict compliance auditor. "
            "You always respond with valid JSON only, no extra text."
        )),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    return ComplianceResult(
        video_id=video_id,
        video_title=video_title,
        status=ComplianceStatus(data["status"]),
        issues=data.get("issues", []),
        recommendations=data.get("recommendations", []),
        summary=data.get("summary", ""),
    )