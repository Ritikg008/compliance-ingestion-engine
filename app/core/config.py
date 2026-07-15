from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    # --- Groq LLM ---
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Qdrant Vector DB ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "compliance_videos"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  

    # --- Whisper (faster-whisper) ---
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"  

    # --- LangSmith Observability ---
    langsmith_api_key: str | None = None
    langsmith_project: str = "compliance-ingestion-engine"
    langchain_tracing_v2: bool = True

    # --- Paths ---
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    temp_video_dir: Path = base_dir / "data" / "temp_video"
    temp_frames_dir: Path = base_dir / "data" / "temp_frames"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

settings.temp_video_dir.mkdir(parents=True, exist_ok=True)
settings.temp_frames_dir.mkdir(parents=True, exist_ok=True)