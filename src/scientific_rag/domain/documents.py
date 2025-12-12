from typing import Any

from pydantic import BaseModel, Field

from scientific_rag.domain.types import DataSource, SectionType


class ScientificPaper(BaseModel):
    paper_id: str = Field(..., description="Unique identifier for the paper")
    abstract: str = Field(..., description="Paper abstract")
    article: str = Field(..., description="Full article text")
    section_names: str = Field(..., description="Section structure")
    source: DataSource = Field(..., description="Data source (arxiv/pubmed)")

    class Config:
        frozen = True


class PaperChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    paper_id: str = Field(..., description="Source paper ID")
    source: DataSource = Field(..., description="Data source")
    section: SectionType = Field(..., description="Section type")
    position: int = Field(..., description="Position in paper (0-indexed)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    score: float | None = Field(None, description="Retrieval score")
    embedding: list[float] | None = Field(None, description="Text embedding vector")

    class Config:
        frozen = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "paper_id": self.paper_id,
            "source": self.source.value,
            "section": self.section.value,
            "position": self.position,
            "metadata": self.metadata,
        }
