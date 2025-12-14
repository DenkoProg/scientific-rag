from pathlib import Path

from loguru import logger
import typer

from scientific_rag.scripts.chunk_data import chunk_data
from scientific_rag.scripts.index_qdrant import index_qdrant


app = typer.Typer(name="scientific-rag", help="Scientific RAG data pipeline CLI", add_completion=False)


@app.command()
def chunk(
    batch_size: int = typer.Option(10000, "--batch-size", "-b", help="Papers per batch"),
) -> None:
    """Process papers and generate chunks."""
    chunk_data(batch_size=batch_size)


@app.command()
def index(
    chunks_file: str = typer.Option(None, "--chunks-file", "-f", help="Path to chunks JSON file"),
    embedding_batch_size: int = typer.Option(32, "--embedding-batch-size", "-eb"),
    upload_batch_size: int = typer.Option(100, "--upload-batch-size", "-ub"),
    create_collection: bool = typer.Option(True, "--create-collection/--no-create-collection"),
    process_batch_size: int = typer.Option(10000, "--process-batch-size", "-pb", help="Process chunks in batches"),
) -> None:
    """Embed chunks and upload to Qdrant."""
    chunks_path = Path(chunks_file) if chunks_file else None
    index_qdrant(
        chunks_file=chunks_path,
        embedding_batch_size=embedding_batch_size,
        upload_batch_size=upload_batch_size,
        create_collection=create_collection,
        process_batch_size=process_batch_size,
    )


@app.command()
def pipeline(
    chunk_batch_size: int = typer.Option(10000, "--chunk-batch-size", "-cb"),
    embedding_batch_size: int = typer.Option(32, "--embedding-batch-size", "-eb"),
    upload_batch_size: int = typer.Option(100, "--upload-batch-size", "-ub"),
    create_collection: bool = typer.Option(True, "--create-collection/--no-create-collection"),
    process_batch_size: int = typer.Option(10000, "--process-batch-size", "-pb", help="Process chunks in batches"),
) -> None:
    """Run complete pipeline: chunk → embed → index."""
    logger.info("Step 1/2: Chunking data")
    chunk_data(batch_size=chunk_batch_size)

    logger.info("Step 2/2: Indexing to Qdrant")
    index_qdrant(
        chunks_file=None,
        embedding_batch_size=embedding_batch_size,
        upload_batch_size=upload_batch_size,
        create_collection=create_collection,
        process_batch_size=process_batch_size,
    )


@app.command()
def info() -> None:
    """Show pipeline configuration and Qdrant status."""
    from scientific_rag.infrastructure.qdrant import QdrantService
    from scientific_rag.settings import settings

    logger.info(f"Dataset: {settings.dataset_name} ({settings.dataset_split})")
    logger.info(f"Chunk: size={settings.chunk_size}, overlap={settings.chunk_overlap}")
    logger.info(f"Embedding: {settings.embedding_model_name}")
    logger.info(f"Qdrant: {settings.qdrant_url} / {settings.qdrant_collection_name}")

    try:
        qdrant = QdrantService()
        info = qdrant.get_collection_info()
        if info["exists"]:
            logger.info(
                f"Collection: {info['points_count']} points, {info['indexed_vectors_count']} indexed vectors, status={info['status']}"
            )
        else:
            logger.warning(f"Collection '{settings.qdrant_collection_name}' does not exist")
    except Exception as e:
        logger.error(f"Qdrant connection failed: {e}")


if __name__ == "__main__":
    app()
