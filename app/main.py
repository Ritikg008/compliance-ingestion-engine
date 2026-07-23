import os
import contextlib
from fastapi import FastAPI
from app.api.routes import router
from app.core.config import settings
from app.core.qdrant_client import init_collection
from app.core.ingestion_tracker import init_tracker_collection

# Set LangSmith env vars for tracing
os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
if settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    init_collection()
    init_tracker_collection()
    print("✅ Compliance Ingestion Engine started.")
    yield
    # Shutdown
    pass

app = FastAPI(
    title="Compliance Ingestion Engine",
    description="Multi-modal AI compliance auditing for YouTube videos using LangGraph + Groq + Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")