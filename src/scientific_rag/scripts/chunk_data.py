import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from scientific_rag.application.chunking.scientific_chunker import ScientificChunker
from scientific_rag.application.data_loader import DataLoader
from scientific_rag.settings import settings


def chunk_data(batch_size: int = 10000):
    output_dir = Path(settings.root_dir) / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading papers from {settings.dataset_name} ({settings.dataset_split} split)")
    data_loader = DataLoader(
        dataset_name=settings.dataset_name,
        split=settings.dataset_split,
        cache_dir=settings.dataset_cache_dir,
    )
    papers = data_loader.load_papers()

    logger.info(f"Loaded {len(papers)} papers")
    logger.info(f"Chunking with size={settings.chunk_size}, overlap={settings.chunk_overlap}, batch_size={batch_size}")

    chunker = ScientificChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size,
    )

    output_file = output_dir / f"chunks_{settings.dataset_split}.json"
    total_chunks = 0

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")

        for batch_idx in range(0, len(papers), batch_size):
            batch_papers = papers[batch_idx : batch_idx + batch_size]
            batch_chunks = []

            for paper in tqdm(batch_papers, desc=f"Batch {batch_idx // batch_size + 1}", leave=False):
                chunks = chunker.chunk(paper)
                batch_chunks.extend(chunks)

            for i, chunk in enumerate(batch_chunks):
                if total_chunks > 0 or i > 0:
                    f.write(",\n")
                json.dump(chunk.model_dump(), f, ensure_ascii=False, indent=2)
                total_chunks += 1

            logger.info(f"Processed batch {batch_idx // batch_size + 1}: {len(batch_chunks)} chunks")

        f.write("\n]")

    logger.success(f"Saved {total_chunks} chunks to {output_file}")

    stats = {
        "total_papers": len(papers),
        "total_chunks": total_chunks,
        "avg_chunks_per_paper": total_chunks / len(papers) if papers else 0,
        "config": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "min_chunk_size": settings.min_chunk_size,
            "batch_size": batch_size,
        },
    }

    stats_file = output_dir / f"stats_{settings.dataset_split}.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Statistics: {stats}")


if __name__ == "__main__":
    chunk_data()
