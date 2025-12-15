from collections.abc import Sequence
from typing import Any

from fastembed import SparseTextEmbedding
from loguru import logger
from qdrant_client import QdrantClient as SyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    NamedSparseVector,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from scientific_rag.domain.documents import PaperChunk
from scientific_rag.domain.queries import QueryFilters
from scientific_rag.settings import settings


class QdrantService:
    def __init__(self) -> None:
        self.url = settings.qdrant_url
        self.api_key = settings.qdrant_api_key
        self.collection_name = settings.qdrant_collection_name

        logger.info(f"Initializing Qdrant client: {self.url}")
        self.client = SyncQdrantClient(url=self.url, api_key=self.api_key, timeout=30)

    def create_collection(self, vector_size: int = 384, distance: Distance = Distance.COSINE) -> None:
        if self.client.collection_exists(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists")
            return

        logger.info(f"Creating collection '{self.collection_name}' with dense and sparse vectors")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=vector_size, distance=distance),
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=True,
                    ),
                    modifier=Modifier.IDF,
                )
            },
        )

        for field in ["source", "section", "paper_id"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema="keyword",
            )
        logger.info(f"Collection '{self.collection_name}' created with indexes")

    def upsert_chunks(self, chunks: list[PaperChunk], sparse_embeddings: list[Any] | None = None) -> int:
        if not chunks:
            return 0

        points = []
        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                continue

            vectors = {"dense": chunk.embedding}

            if sparse_embeddings and i < len(sparse_embeddings):
                sparse = sparse_embeddings[i]
                vectors["bm25"] = SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())

            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vectors,
                    payload=chunk.to_dict(),
                )
            )

        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        logger.info(f"Uploaded {len(points)} chunks to Qdrant")
        return len(points)

    def search_dense(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: QueryFilters | None = None,
    ) -> list[PaperChunk]:
        """Standard semantic search using dense vectors."""
        return self._execute_search(
            vector_name="dense",
            vector_data=query_vector,
            is_sparse=False,
            limit=limit,
            filters=filters,
        )

    def search_sparse(
        self,
        query_sparse_indices: list[int],
        query_sparse_values: list[float],
        limit: int = 10,
        filters: QueryFilters | None = None,
    ) -> list[PaperChunk]:
        """BM25-style search using sparse vectors."""
        sparse_vec = SparseVector(
            indices=query_sparse_indices,
            values=query_sparse_values,
        )

        return self._execute_search(
            vector_name="bm25",
            vector_data=sparse_vec,
            is_sparse=True,
            limit=limit,
            filters=filters,
        )

    def _execute_search(
        self,
        vector_name: str,
        vector_data: Any,
        is_sparse: bool,
        limit: int,
        filters: QueryFilters | None,
    ) -> list[PaperChunk]:
        query_filter = self._build_filters(filters) if filters else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            using=vector_name,
            query=vector_data,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            PaperChunk(
                **hit.payload,
                score=hit.score,
                embedding=None,
            )
            for hit in results.points
        ]

    def _build_filters(self, filters: QueryFilters) -> Filter | None:
        filter_dict = filters.to_qdrant_filter()
        if not filter_dict:
            return None

        must_conditions = []
        target_list = filter_dict.get("must", []) if "must" in filter_dict else [filter_dict]

        for item in target_list:
            if "key" in item and "match" in item:
                must_conditions.append(FieldCondition(key=item["key"], match=MatchValue(value=item["match"]["value"])))

        return Filter(must=must_conditions) if must_conditions else None

    def get_collection_info(self) -> dict[str, Any]:
        if not self.client.collection_exists(self.collection_name):
            return {"exists": False}
        info = self.client.get_collection(self.collection_name)
        return {"exists": True, "points_count": info.points_count}

    def close(self):
        self.client.close()
        logger.info("Qdrant client closed")


qdrant_service = QdrantService()
