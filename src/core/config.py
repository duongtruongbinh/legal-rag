"""Application configuration and environment variables."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Pydantic automatically maps uppercase environment variables to 
    lowercase attributes (e.g., GOOGLE_API_KEY -> google_api_key).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,  # Ensures GOOGLE_API_KEY matches google_api_key
    )

    # --- API Configuration ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # --- Qdrant Configuration ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legal_hybrid_v2"

    # --- Embedding Models ---
    dense_model: str = "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"
    sparse_model: str = "Qdrant/bm25"
    embedding_device: str = "cuda"

    # --- LLM Configuration ---
    # REQUIRED: Must be set in .env or environment
    google_api_key: str 
    llm_model: str = "gemini-2.5-flash-lite"
    llm_temperature: float = 0.1

    # --- Retrieval Configuration ---
    retrieval_top_k: int = 20
    rerank_top_n: int = 5

    # --- Storage Paths ---
    data_dir: Path = Path("./data")
    docstore_path: Path = Path("./data/docstore")

    def model_post_init(self, __context):
        """Create necessary directories after initialization."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.docstore_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance to avoid re-reading .env file."""
    return Settings()


# Singleton instance
settings = get_settings()