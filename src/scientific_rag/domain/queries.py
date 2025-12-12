from typing import Literal

from pydantic import BaseModel, Field


class QueryFilters(BaseModel):
    source: Literal["arxiv", "pubmed", "any"] = Field(default="any", description="Filter by data source")
    section: Literal["introduction", "methods", "results", "conclusion", "any"] = Field(
        default="any", description="Filter by section type"
    )

    def to_qdrant_filter(self) -> dict | None:
        conditions = []

        if self.source != "any":
            conditions.append({"key": "source", "match": {"value": self.source}})

        if self.section != "any":
            conditions.append({"key": "section", "match": {"value": self.section}})

        if not conditions:
            return None

        return {"must": conditions} if len(conditions) > 1 else conditions[0]


class Query(BaseModel):
    text: str = Field(..., description="Query text")
    filters: QueryFilters | None = Field(None, description="Optional metadata filters")
    top_k: int = Field(default=10, description="Number of results to retrieve")


class ExpandedQuery(BaseModel):
    original: str = Field(..., description="Original query text")
    variations: list[str] = Field(default_factory=list, description="Query variations")
    filters: QueryFilters | None = Field(None, description="Metadata filters")

    def all_queries(self) -> list[str]:
        return [self.original] + self.variations


class EmbeddedQuery(BaseModel):
    text: str = Field(..., description="Query text")
    embedding: list[float] = Field(..., description="Query embedding vector")
    filters: QueryFilters | None = Field(None, description="Metadata filters")
