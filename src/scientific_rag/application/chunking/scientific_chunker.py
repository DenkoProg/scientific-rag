import hashlib
import re

from scientific_rag.domain.documents import PaperChunk, ScientificPaper
from scientific_rag.domain.types import SectionType
from scientific_rag.settings import settings


class ScientificChunker:
    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        min_chunk_size: int = settings.min_chunk_size,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, paper: ScientificPaper) -> list[PaperChunk]:
        chunks: list[PaperChunk] = []
        position = 0

        if paper.abstract:
            abstract_text = self._normalize_placeholders(paper.abstract or "")
            chunk_id = self._generate_chunk_id(paper.paper_id, position)
            chunks.append(
                PaperChunk(
                    chunk_id=chunk_id,
                    text=abstract_text.strip(),
                    paper_id=paper.paper_id,
                    source=paper.source,
                    section=SectionType.ABSTRACT,
                    position=position,
                )
            )
            position += 1

        if paper.article:
            article_text = self._normalize_placeholders(paper.article or "")
            paragraphs = self._get_paragraphs(article_text)
            article_chunks = self._group_paragraphs(paragraphs)

            for idx, chunk_text in enumerate(article_chunks):
                word_count = len(chunk_text.split())
                if word_count < self.min_chunk_size:
                    continue

                section_type = self._infer_section_by_position(idx, len(article_chunks))
                chunk_id = self._generate_chunk_id(paper.paper_id, position)
                chunks.append(
                    PaperChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        paper_id=paper.paper_id,
                        source=paper.source,
                        section=section_type,
                        position=position,
                    )
                )
                position += 1

        return chunks

    def _get_paragraphs(self, text: str) -> list[str]:
        paragraphs = []
        for p in text.split("\n"):
            p = p.strip()
            if p:
                paragraphs.append(p)
        return paragraphs

    def _group_paragraphs(self, paragraphs: list[str]) -> list[str]:
        chunks: list[str] = []
        current_paragraphs: list[str] = []
        current_word_count = 0

        for para in paragraphs:
            para_word_count = len(para.split())

            if current_word_count + para_word_count > self.chunk_size and current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))

                overlap_paragraphs: list[str] = []
                overlap_word_count = 0
                for p in reversed(current_paragraphs):
                    p_words = len(p.split())
                    if overlap_word_count + p_words <= self.chunk_overlap:
                        overlap_paragraphs.insert(0, p)
                        overlap_word_count += p_words
                    else:
                        break

                current_paragraphs = overlap_paragraphs
                current_word_count = overlap_word_count

            current_paragraphs.append(para)
            current_word_count += para_word_count

        if current_paragraphs:
            chunks.append("\n\n".join(current_paragraphs))

        return chunks

    def _infer_section_by_position(self, chunk_idx: int, total_chunks: int) -> SectionType:
        if total_chunks <= 1:
            return SectionType.OTHER

        relative_position = chunk_idx / total_chunks

        if relative_position < 0.15:
            return SectionType.INTRODUCTION
        elif relative_position < 0.5:
            return SectionType.METHODS
        elif relative_position < 0.85:
            return SectionType.RESULTS
        else:
            return SectionType.CONCLUSION

    def _generate_chunk_id(self, paper_id: str, position: int) -> str:
        content = f"{paper_id}_{position}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _normalize_placeholders(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"@xmath\d+", "<MATH>", text)
        text = re.sub(r"@xcite", "[CITE]", text)
        text = re.sub(r"\[[^\]]+\]", "<BRACKET>", text)
        text = re.sub(r"(?:<BRACKET>\s*){2,}", "<BRACKET>", text)
        return text.strip()
