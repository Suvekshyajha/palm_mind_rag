from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings come from the .env file (names are not case sensitive)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Groq
    groq_api_key: str = ""
    groq_model: str = ""

    # Databases
    database_url: str
    redis_url: str
    qdrant_url: str
    qdrant_collection: str = "documents"

    # Models
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Ingestion
    max_upload_mb: int = 10
    enable_ocr: bool = False
    upload_dir: str = "storage/uploads"

    # Chunking
    recursive_chunk_size: int = 1000
    recursive_chunk_overlap: int = 150
    semantic_max_chars: int = 1000
    semantic_min_chars: int = 200
    semantic_break_percentile: int = 25

    # Retrieval
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 20
    final_top_k: int = 5
    relevance_threshold: float = 0.3

    # Memory
    history_messages_in_prompt: int = 6
    history_messages_stored: int = 20
    chat_ttl_seconds: int = 86400
    booking_ttl_seconds: int = 3600


@lru_cache
def get_settings() -> Settings:
    """Create the settings once and reuse them."""
    return Settings()