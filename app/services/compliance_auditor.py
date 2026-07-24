from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import settings
from app.models.schemas import DocumentChunk, ComplianceResult, ComplianceStatus
from app.services.compliance_rules import check_compliance_rules, format_violations_for_llm
from typing import List
import json


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
        max_tokens=1024,
    )


def build_audit_prompt(
    chunks: List[DocumentChunk],
    rule_violations_str: str,
) -> str:
    content_blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{chunk.source_type.upper()}]"
        time_info = f" @ {chunk.start_time}s" if chunk.start_time else ""
        content_blocks.append(f"{i}. {source}{time_info}\n{chunk.text}")

    content_str = "\n\n".join(content_blocks)

    return f"""You are a senior compliance auditor reviewing video content.

RULE-BASED PRE-SCREENING RESULTS:
{rule_violations_str}

EXTRACTED CONTENT TO AUDIT:
{content_str}

Based on both the pre-screening results and your own analysis, identify:
- Unsubstantiated health or medical claims
- Misleading financial claims or guarantees
- False advertising or exaggerated product claims
- Missing required disclaimers
- Potentially illegal or deceptive statements
- Privacy or data protection concerns

Respond ONLY with a valid JSON object:
{{
    "status": "compliant" | "non_compliant" | "needs_review",
    "issues": ["issue 1", "issue 2"],
    "recommendations": ["recommendation 1", "recommendation 2"],
    "summary": "one paragraph summary"
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
            issues=["No content could be extracted from this video."],
            recommendations=["Verify the video has audible speech or visible text."],
            summary="Insufficient content was extracted to perform a compliance audit.",
        )

    # Step 1: Run fast regex rules across all chunk text
    full_text = " ".join([c.text for c in chunks])
    violations = check_compliance_rules(full_text)
    rule_violations_str = format_violations_for_llm(violations)

    print(f"Rule-based violations found: {len(violations)}")

    # Step 2: Pass both raw content AND rule findings to LLM
    llm = get_llm()
    prompt = build_audit_prompt(chunks, rule_violations_str)

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