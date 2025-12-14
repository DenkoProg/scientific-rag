from typing import Literal

from loguru import logger
from sentence_transformers import SentenceTransformer

from scientific_rag.settings import settings


class EmbeddingEncoder:
    def __init__(self) -> None:
        self.model_name = settings.embedding_model_name
        self.device = settings.embedding_device
        self.batch_size = settings.embedding_batch_size

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        mode: Literal["query", "passage"],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[list[float]]:
        batch_size = batch_size or self.batch_size

        # specific prefixing for E5/BGE models
        prefix = "query: " if mode == "query" else "passage: "
        prefixed_texts = [f"{prefix}{t}" for t in texts]

        embeddings = self._model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()


encoder = EmbeddingEncoder()
