import re

from loguru import logger
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
from rank_bm25 import BM25Okapi

from scientific_rag.domain.documents import PaperChunk
from scientific_rag.domain.queries import QueryFilters


class BM25Retriever:
    def __init__(self, chunks: list[PaperChunk]):
        self.chunks = chunks
        self._ensure_nltk_resources()

        self.stemmer = SnowballStemmer("english")
        self.stop_words = set(stopwords.words("english"))

        logger.info(f"Preprocessing and indexing {len(chunks)} chunks for BM25...")
        self.corpus = [self._preprocess(chunk.text) for chunk in chunks]

        self.bm25 = BM25Okapi(self.corpus)
        logger.info("BM25 index initialized successfully")

    def _ensure_nltk_resources(self):
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download("stopwords")

    def _preprocess(self, text: str) -> list[str]:
        # 1. Lowercase
        text = text.lower()

        # 2. Remove citations like [1], [12] which are common in scientific papers
        text = re.sub(r"\[\d+\]", "", text)

        # 3. Replace non-alphanumeric chars with spaces (preserves hyphenated words as separate tokens)
        # e.g., "state-of-the-art" -> "state of the art"
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # 4. Simple whitespace split is sufficient after regex cleaning
        tokens = text.split()

        # 5. & 6. Stopword removal and Stemming
        # processing in a list comp is faster than a loop
        clean_tokens = [
            self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 1
        ]

        return clean_tokens

    def search(self, query: str, k: int = 10, filters: QueryFilters | None = None) -> list[PaperChunk]:
        tokenized_query = self._preprocess(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        sorted_indices = np.argsort(scores)[::-1]

        results = []
        raw_scores = []

        for idx in sorted_indices:
            if scores[idx] <= 0:
                break

            chunk = self.chunks[idx]

            if not self._matches_filters(chunk, filters):
                continue

            results.append(chunk)
            raw_scores.append(scores[idx])

            if len(results) >= k:
                break

        if raw_scores:
            max_s = max(raw_scores)
            min_s = min(raw_scores)

            for i, chunk in enumerate(results):
                if max_s == min_s:
                    norm_score = 1.0 if max_s > 0 else 0.0
                else:
                    norm_score = (raw_scores[i] - min_s) / (max_s - min_s)

                results[i] = chunk.model_copy()
                results[i].score = float(norm_score)

        return results

    def _matches_filters(self, chunk: PaperChunk, filters: QueryFilters | None) -> bool:
        if not filters:
            return True
        if filters.source != "any" and chunk.source.value != filters.source:
            return False
        if filters.section != "any" and chunk.section.value != filters.section:
            return False
        return True
