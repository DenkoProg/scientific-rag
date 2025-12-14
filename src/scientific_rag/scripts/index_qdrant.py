from collections.abc import Iterator
import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from scientific_rag.application.embeddings.encoder import encoder
from scientific_rag.domain.documents import PaperChunk
from scientific_rag.infrastructure.qdrant import QdrantService
from scientific_rag.settings import settings


def load_chunks_generator(chunks_file: Path, batch_size: int = 10000) -> Iterator[list[PaperChunk]]:
    logger.info(f"Loading chunks from {chunks_file} in batches of {batch_size}")

    with open(chunks_file, encoding="utf-8") as f:
        chunks_data = json.load(f)

    total_chunks = len(chunks_data)
    logger.info(f"Found {total_chunks} chunks in file")

    for i in range(0, total_chunks, batch_size):
        batch_data = chunks_data[i : i + batch_size]
        batch_chunks = [PaperChunk(**chunk_data) for chunk_data in batch_data]
        yield batch_chunks

    del chunks_data


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
    show_progress: bool = True,
) -> int:
    """Upload chunks to Qdrant in batches."""
    total_uploaded = 0

    iterator = tqdm(range(0, len(chunks), batch_size), desc="Uploading to Qdrant", disable=not show_progress)
    for i in iterator:
        batch = chunks[i : i + batch_size]
        uploaded = qdrant_service.upsert_chunks(batch)
        total_uploaded += uploaded

    return total_uploaded


def index_qdrant(
    chunks_file: Path | str | None = None,
    embedding_batch_size: int = 32,
    upload_batch_size: int = 100,
    create_collection: bool = True,
    process_batch_size: int = 10000,
) -> dict[str, int]:
    """Complete pipeline to index chunks to Qdrant.

    Args:
        chunks_file: Path to chunks JSON file
        embedding_batch_size: Batch size for embedding generation
        upload_batch_size: Batch size for Qdrant upload
        create_collection: Whether to create the collection
        process_batch_size: Process chunks in batches of this size to manage memory
    """
    if chunks_file is None:
        chunks_file = Path(settings.root_dir) / "data" / "processed" / f"chunks_{settings.dataset_split}.json"
    else:
        chunks_file = Path(chunks_file)

    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    qdrant_service = QdrantService()
    if create_collection:
        qdrant_service.create_collection(vector_size=encoder.embedding_dim)

    logger.info("Processing chunks in streaming batches to manage memory...")
    total_uploaded = 0
    batch_num = 0

    for batch_chunks in load_chunks_generator(chunks_file, batch_size=process_batch_size):
        batch_num += 1
        batch_start = (batch_num - 1) * process_batch_size
        batch_end = batch_start + len(batch_chunks)

        logger.info(f"Batch {batch_num}: Embedding chunks {batch_start}-{batch_end} ({len(batch_chunks)} chunks)...")
        batch_chunks = embed_chunks(
            chunks=batch_chunks,
            batch_size=embedding_batch_size,
            show_progress=True,
        )

        logger.info(f"Batch {batch_num}: Uploading chunks {batch_start}-{batch_end} to Qdrant...")
        batch_uploaded = index_chunks_to_qdrant(
            chunks=batch_chunks,
            qdrant_service=qdrant_service,
            batch_size=upload_batch_size,
            show_progress=True,
        )
        total_uploaded += batch_uploaded

        logger.success(f"Batch {batch_num} complete: {batch_uploaded} chunks uploaded (Total: {total_uploaded})")

    logger.info("Getting final statistics...")
    collection_info = qdrant_service.get_collection_info()
    stats = {
        "chunks_uploaded": total_uploaded,
        "collection_points": collection_info.get("points_count", 0),
        "collection_vectors": collection_info.get("index_vectors_count", 0),
    }

    logger.success(f"Indexing complete: {stats}")
    return stats


if __name__ == "__main__":
    index_qdrant()
