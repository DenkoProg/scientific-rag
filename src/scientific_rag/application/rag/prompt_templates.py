from scientific_rag.domain.documents import PaperChunk


class RAGPrompts:
    SYSTEM_PROMPT = """You are an expert scientific research assistant.
Your task is to answer the user's question accurately using ONLY the provided context snippets.

Rules:
1. Use information ONLY from the provided context. If the answer is not in the context, say "I cannot answer this based on the provided papers."
2. Cite your sources for every specific claim using square brackets like [1], [2].
3. Do not use outside knowledge unless it's for general definitions or connecting concepts found in the text.
4. Be concise and professional.
5. If multiple sources discuss the same point, cite all of them, e.g., [1][3].
"""

    @staticmethod
    def format_context(chunks: list[PaperChunk]) -> str:
        formatted_parts = []
        for i, chunk in enumerate(chunks, 1):
            # We include metadata to help the LLM understand the source
            source_label = chunk.source.value.upper()
            section_label = chunk.section.value

            part = (
                f"[{i}] Source: {source_label} | Paper ID: {chunk.paper_id} | Section: {section_label}\n"
                f"Content: {chunk.text}"
            )
            formatted_parts.append(part)

        return "\n\n".join(formatted_parts)

    @staticmethod
    def generate_rag_prompt(query: str, chunks: list[PaperChunk]) -> str:
        context_str = RAGPrompts.format_context(chunks)

        return f"""Please answer the question based on the context below.

        --- CONTEXT START ---
        {context_str}
        --- CONTEXT END ---

        Question: {query}

        Answer:"""
