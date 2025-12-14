from typing import Any

from loguru import logger
from qdrant_client import QdrantClient as SyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

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

        logger.info(f"Creating collection '{self.collection_name}' with vector size {vector_size}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

        for field in ["source", "section", "paper_id"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema="keyword",
            )
        logger.info(f"Collection '{self.collection_name}' created with indexes")

    def get_collection_info(self) -> dict[str, Any]:
        if not self.client.collection_exists(self.collection_name):
            return {"exists": False}

        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }

    def upsert_chunks(self, chunks: list[PaperChunk]) -> int:
        if not chunks:
            return 0

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload=chunk.to_dict(),
            )
            for chunk in chunks
            if chunk.embedding is not None
        ]

        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        logger.info(f"Uploaded {len(points)} chunks to Qdrant")
        return len(points)

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        filters: QueryFilters | None = None,
    ) -> list[PaperChunk]:
        query_filter = self._build_filters(filters) if filters else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
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

    def close(self) -> None:
        self.client.close()
        logger.info("Qdrant client closed")


qdrant_service = QdrantService()
