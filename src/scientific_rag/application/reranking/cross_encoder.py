from loguru import logger
from sentence_transformers import CrossEncoder

from scientific_rag.domain.documents import PaperChunk
from scientific_rag.settings import settings


class CrossEncoderReranker:
    def __init__(self):
        self.model_name = settings.reranker_model_name
        self.device = settings.embedding_device

        logger.info(f"Loading Reranker model: {self.model_name} on {self.device}")
        self.model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Reranker model loaded")

    def rerank(self, query: str, chunks: list[PaperChunk], top_k: int = settings.rerank_top_k) -> list[PaperChunk]:
        if not chunks:
            return []

        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)

        reranked_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.model_copy()
            chunk_copy.score = float(scores[i])
            reranked_chunks.append(chunk_copy)

        reranked_chunks.sort(key=lambda x: x.score, reverse=True)

        return reranked_chunks[:top_k]


reranker = CrossEncoderReranker()
