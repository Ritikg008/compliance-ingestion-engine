import os
from fastapi import FastAPI
from app.api.routes import router
from app.core.config import settings
from app.core.qdrant_client import init_collection

# Set LangSmith env vars for tracing
os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
if settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key

app = FastAPI(
    title="Compliance Ingestion Engine",
    description="Multi-modal AI compliance auditing for YouTube videos using LangGraph + Groq + Qdrant",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    """Ensures Qdrant collection exists before accepting requests."""
    init_collection()
    print("Compliance Ingestion Engine started.")

app.include_router(router, prefix="/api/v1")