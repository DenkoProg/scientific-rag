from loguru import logger

from scientific_rag.application.retrieval.bm25_retriever import BM25Retriever
from scientific_rag.application.retrieval.dense_retriever import DenseRetriever
from scientific_rag.domain.documents import PaperChunk
from scientific_rag.domain.queries import QueryFilters
from scientific_rag.settings import settings


class HybridRetriever:
    def __init__(self):
        logger.info("Initializing Hybrid Retriever...")
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        logger.info("Hybrid Retriever ready.")

    def search(
        self,
        query: str,
        k: int = settings.retrieval_top_k,
        filters: QueryFilters | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        bm25_weight: float = settings.bm25_weight,
        dense_weight: float = settings.dense_weight,
    ) -> list[PaperChunk]:
        """
        Perform hybrid search with filters applied at the source.
        """
        chunk_map: dict[str, PaperChunk] = {}
        scores: dict[str, float] = {}

        if use_bm25:
            bm25_results = self.bm25.search(query, k=k, filters=filters)

            for chunk in bm25_results:
                chunk_id = chunk.chunk_id
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk
                    scores[chunk_id] = 0.0
                scores[chunk_id] += chunk.score * bm25_weight

        if use_dense:
            dense_results = self.dense.search(query, k=k, filters=filters)

            for chunk in dense_results:
                chunk_id = chunk.chunk_id
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk
                    scores[chunk_id] = 0.0
                scores[chunk_id] += chunk.score * dense_weight

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        final_results = []
        for chunk_id in sorted_ids[:k]:
            chunk = chunk_map[chunk_id]
            result_chunk = chunk.model_copy()
            result_chunk.score = scores[chunk_id]
            final_results.append(result_chunk)

        return final_results
