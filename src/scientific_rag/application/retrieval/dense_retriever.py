from loguru import logger

from scientific_rag.application.embeddings.encoder import encoder
from scientific_rag.domain.documents import PaperChunk
from scientific_rag.domain.queries import QueryFilters
from scientific_rag.infrastructure.qdrant import qdrant_service


class DenseRetriever:
    def __init__(self):
        info = qdrant_service.get_collection_info()
        if not info["exists"]:
            logger.warning("Qdrant collection does not exist! Run 'make index-qdrant' first to populate data.")

    def search(self, query: str, k: int = 10, filters: QueryFilters | None = None) -> list[PaperChunk]:
        try:
            query_vector = encoder.encode([query], mode="query")[0]
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        results = qdrant_service.search_dense(query_vector=query_vector, limit=k, filters=filters)

        return results
