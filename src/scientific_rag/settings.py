import logging
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Application
    root_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    logging_level: int = logging.INFO

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "scientific_papers"

    # Embeddings
    embedding_model_name: str = "intfloat/e5-small-v2"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"  # or "cuda"

    # Chunking
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 50
    min_chunk_size: int = 100

    # Retrieval
    retrieval_top_k: int = 10
    bm25_weight: float = 0.5
    dense_weight: float = 0.5

    # Reranking
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_top_k: int = 5

    # LLM
    llm_provider: str = "openrouter"
    llm_model: str = "openai/gpt-oss-120b:free"
    llm_api_key: str | None = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # Data
    dataset_name: str = "armanc/scientific_papers"
    dataset_split: str = "arxiv"  # arxiv, pubmed
    dataset_cache_dir: str = Field(default_factory=lambda: str(Path(__file__).parent.parent.parent / "data" / "raw"))
    dataset_sample_size: int | None = None  # None for full dataset


settings = AppSettings()
