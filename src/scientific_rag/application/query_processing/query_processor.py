import re

from loguru import logger

from scientific_rag.application.rag.llm_client import llm_client
from scientific_rag.domain.queries import ExpandedQuery, QueryFilters
from scientific_rag.settings import settings


class QueryProcessor:
    def __init__(self, expand_to_n: int = 3):
        self.expand_to_n = expand_to_n

    def process(self, query: str, use_expansion: bool = True, extract_filters: bool = True) -> ExpandedQuery:
        if extract_filters:
            cleaned_query, filters = self._extract_filters(query)
        else:
            cleaned_query = query
            filters = QueryFilters(source="any", section="any")

        variations = []
        if use_expansion and self.expand_to_n > 1:
            variations = self._expand_query(cleaned_query)

        logger.info(
            f"Processed query: '{query}' -> '{cleaned_query}' | "
            f"Filters: {filters} | Expansion: {len(variations)} vars"
        )

        return ExpandedQuery(original=cleaned_query, variations=variations, filters=filters)

    def _extract_filters(self, query: str) -> tuple[str, QueryFilters]:
        source = "any"
        section = "any"

        query_lower = query.lower()
        cleaned_query = query

        if "arxiv" in query_lower:
            source = "arxiv"
            cleaned_query = re.sub(
                r"\b(from|in|on)?\s*arxiv\s*(papers|articles)?\b", "", cleaned_query, flags=re.IGNORECASE
            )
        elif "pubmed" in query_lower:
            source = "pubmed"
            cleaned_query = re.sub(
                r"\b(from|in|on)?\s*pubmed\s*(papers|articles)?\b", "", cleaned_query, flags=re.IGNORECASE
            )

        section_patterns = {
            "introduction": [r"introduction", r"intro"],
            "methods": [r"methods", r"methodology", r"experiment setup"],
            "results": [r"results", r"findings", r"performance"],
            "conclusion": [r"conclusion", r"summary", r"discussion"],
        }

        found_section = False
        for sec_name, patterns in section_patterns.items():
            if found_section:
                break
            for pattern in patterns:
                full_pattern = rf"\b(in|from|check|read)?\s*(the)?\s*{pattern}\s*(section)?\b"
                if re.search(full_pattern, query_lower):
                    section = sec_name
                    cleaned_query = re.sub(full_pattern, "", cleaned_query, flags=re.IGNORECASE)
                    found_section = True
                    break

        cleaned_query = " ".join(cleaned_query.split())
        if not cleaned_query:
            cleaned_query = query

        return cleaned_query, QueryFilters(source=source, section=section)

    def _expand_query(self, query: str) -> list[str]:
        if not settings.llm_api_key:
            logger.warning("No LLM API Key set. Skipping expansion.")
            return []

        prompt = f"""
        Generate {self.expand_to_n - 1} different search queries for a scientific database based on the input.
        The variations should capture the same technical intent but use alternative terminology or keywords.

        Output ONLY the variations, separated by "###". Do not number them.

        Input: {query}
        """

        try:
            content = llm_client.generate(prompt=prompt)
            variations = [v.strip() for v in content.split("###") if v.strip()]
            final_variations = [v for v in variations if v.lower() != query.lower()]
            return final_variations[: self.expand_to_n - 1]

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []
