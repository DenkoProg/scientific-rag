from pathlib import Path

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from scientific_rag.domain.documents import ScientificPaper
from scientific_rag.domain.types import DataSource
from scientific_rag.settings import settings


class DataLoader:
    def __init__(
        self,
        dataset_name: str | None = None,
        split: str | None = None,
        cache_dir: str | None = None,
    ):
        """Initialize data loader.

        Args:
            dataset_name: HuggingFace dataset name (default from settings)
            split: Dataset split - "arxiv", "pubmed", or both (default from settings)
            cache_dir: Cache directory for downloaded data (default from settings)
        """
        self.dataset_name = dataset_name or settings.dataset_name
        self.split = split or settings.dataset_split
        self.cache_dir = str(cache_dir or settings.dataset_cache_dir)

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def load_papers(
        self,
        sample_size: int | None = None,
        data_split: str = "train",
    ) -> list[ScientificPaper]:
        """Load scientific papers from dataset.

        Args:
            sample_size: Number of papers to load (None for all)
            data_split: Data split - "train", "validation", or "test"

        Returns:
            List of ScientificPaper objects
        """
        sample_size = sample_size or settings.dataset_sample_size

        logger.info(f"Loading {self.split} papers from {self.dataset_name} ({data_split} split)")

        dataset = load_dataset(
            self.dataset_name,
            self.split,
            split=data_split,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        if sample_size is not None:
            logger.info(f"Sampling {sample_size} papers")
            dataset = dataset.select(range(min(sample_size, len(dataset))))

        papers = []
        for idx, item in enumerate(tqdm(dataset, desc="Loading papers")):
            try:
                paper = ScientificPaper(
                    paper_id=f"{self.split}_{idx}",
                    abstract=item.get("abstract", ""),
                    article=item.get("article", ""),
                    section_names=item.get("section_names", ""),
                    source=DataSource.ARXIV if self.split == "arxiv" else DataSource.PUBMED,
                )
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse paper {idx}: {e}")
                continue

        logger.info(f"Loaded {len(papers)} papers")
        return papers

    def load_both_sources(
        self,
        sample_size_per_source: int | None = None,
        data_split: str = "train",
    ) -> list[ScientificPaper]:
        """Load papers from both ArXiv and PubMed.

        Args:
            sample_size_per_source: Number of papers per source
            data_split: Data split - "train", "validation", or "test"

        Returns:
            Combined list of papers from both sources
        """
        papers = []

        arxiv_loader = DataLoader(
            dataset_name=self.dataset_name,
            split="arxiv",
            cache_dir=self.cache_dir,
        )
        papers.extend(arxiv_loader.load_papers(sample_size_per_source, data_split))

        pubmed_loader = DataLoader(
            dataset_name=self.dataset_name,
            split="pubmed",
            cache_dir=self.cache_dir,
        )
        papers.extend(pubmed_loader.load_papers(sample_size_per_source, data_split))

        logger.info(f"Loaded {len(papers)} papers from both sources")
        return papers
