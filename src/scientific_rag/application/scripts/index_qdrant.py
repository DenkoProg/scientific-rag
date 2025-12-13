import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from scientific_rag.application.embeddings.encoder import encoder
from scientific_rag.domain.documents import PaperChunk
from scientific_rag.infrastructure.qdrant import QdrantService
from scientific_rag.settings import settings


def load_chunks(chunks_file: Path) -> list[PaperChunk]:
    """Load chunks from JSON file."""
    logger.info(f"Loading chunks from {chunks_file}")

    with open(chunks_file, encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = [PaperChunk(**chunk_data) for chunk_data in chunks_data]
    logger.info(f"Loaded {len(chunks)} chunks")

    return chunks


def embed_chunks(
    chunks: list[PaperChunk],
    batch_size: int = 32,
    show_progress: bool = True,
) -> list[PaperChunk]:
    """Embed chunks using the encoder."""
    logger.info(f"Embedding {len(chunks)} chunks with batch size {batch_size}")

    texts = [chunk.text for chunk in chunks]

    embeddings = encoder.encode(
        texts=texts,
        mode="passage",
        batch_size=batch_size,
        show_progress=show_progress,
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    logger.success(f"Embedded {len(chunks)} chunks")
    return chunks


def index_chunks_to_qdrant(
    chunks: list[PaperChunk],
    qdrant_service: QdrantService,
    batch_size: int = 100,
) -> int:
    """Upload chunks to Qdrant in batches."""
    logger.info(f"Indexing {len(chunks)} chunks to Qdrant")

    total_uploaded = 0

    for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading to Qdrant"):
        batch = chunks[i : i + batch_size]
        uploaded = qdrant_service.upsert_chunks(batch)
        total_uploaded += uploaded

    logger.success(f"Indexed {total_uploaded} chunks to Qdrant")
    return total_uploaded


def index_qdrant(
    chunks_file: Path | str | None = None,
    embedding_batch_size: int = 32,
    upload_batch_size: int = 100,
    create_collection: bool = True,
) -> dict[str, int]:
    """Complete pipeline to index chunks to Qdrant."""
    if chunks_file is None:
        chunks_file = Path(settings.root_dir) / "data" / "processed" / f"chunks_{settings.dataset_split}.json"
    else:
        chunks_file = Path(chunks_file)

    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    qdrant_service = QdrantService()
    if create_collection:
        qdrant_service.create_collection(vector_size=encoder.embedding_dim)

    chunks = load_chunks(chunks_file)
    chunks = embed_chunks(
        chunks=chunks,
        batch_size=embedding_batch_size,
        show_progress=True,
    )
    total_uploaded = index_chunks_to_qdrant(
        chunks=chunks,
        qdrant_service=qdrant_service,
        batch_size=upload_batch_size,
    )

    collection_info = qdrant_service.get_collection_info()
    stats = {
        "chunks_loaded": len(chunks),
        "chunks_uploaded": total_uploaded,
        "collection_points": collection_info.get("points_count", 0),
        "collection_vectors": collection_info.get("index_vectors_count", 0),
    }

    logger.info(f"Indexing complete: {stats}")
    return stats


if __name__ == "__main__":
    index_qdrant()
