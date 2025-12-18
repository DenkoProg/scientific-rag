from typing import Any

from pydantic import BaseModel, Field

from scientific_rag.domain.documents import PaperChunk


class RAGResponse(BaseModel):
    """Structured response from the RAG pipeline."""

    answer: str
    original_query: str
    generated_query_variations: list[str] = []
    retrieved_chunks: list[PaperChunk]
    total_chunks_retrieved: int = 0  # Total chunks before reranking
    used_filters: dict[str, Any] | None = None
    execution_time: float
