"""Application configuration and environment variables."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- API Configuration ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # --- Qdrant Configuration ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legal_hybrid_v3"

    # --- Embedding Models ---
    dense_model: str = "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"
    sparse_model: str = "Qdrant/bm25"
    embedding_device: str = "cuda"

    # --- Reranker Configuration ---
    reranker_model: str = "namdp-ptit/ViRanker"
    reranker_device: str = "cuda"
    reranker_top_n: int = 5

    # --- LLM Configuration ---
    google_api_key: str
    llm_model: str = "gemini-2.5-flash-lite"
    llm_temperature: float = 0.1

    # --- Retrieval Configuration ---
    retrieval_top_k: int = 30  # Initial retrieval before reranking

    # --- Text Splitting Configuration ---
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 512
    child_chunk_overlap: int = 100

    # --- Template Path ---
    templates_dir: Path = Path(__file__).parent.parent / "templates"

    # --- Storage Paths ---
    data_dir: Path = Path("./data")

    def model_post_init(self, __context):
        """Create necessary directories after initialization."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
