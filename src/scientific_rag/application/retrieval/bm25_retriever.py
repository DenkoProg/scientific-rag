from fastembed import SparseTextEmbedding
from loguru import logger

from scientific_rag.domain.documents import PaperChunk
from scientific_rag.domain.queries import QueryFilters
from scientific_rag.infrastructure.qdrant import qdrant_service
from scientific_rag.settings import settings


class BM25Retriever:
    def __init__(self):
        logger.info(f"Initializing BM25 Retriever with model: {settings.sparse_embedding_model_name}")
        self.sparse_encoder = SparseTextEmbedding(model_name=settings.sparse_embedding_model_name)

    def search(self, query: str, k: int = 10, filters: QueryFilters | None = None) -> list[PaperChunk]:
        query_sparse = next(self.sparse_encoder.embed([query]))

        results = qdrant_service.search_sparse(
            query_sparse_indices=query_sparse.indices.tolist(),
            query_sparse_values=query_sparse.values.tolist(),
            limit=k,
            filters=filters,
        )

        if results:
            scores = [r.score for r in results]
            max_s = max(scores)
            min_s = min(scores)
            for r in results:
                if max_s > min_s:
                    r.score = (r.score - min_s) / (max_s - min_s)
                else:
                    r.score = 1.0 if max_s > 0 else 0.0

        return results
