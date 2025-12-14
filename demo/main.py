from pathlib import Path
import sys
from typing import Any

import gradio as gr


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scientific_rag.domain.documents import PaperChunk
from scientific_rag.domain.queries import Query, QueryFilters
from scientific_rag.domain.types import DataSource, SectionType


# =============================================================================
# Service Description
# =============================================================================

MAIN_HEADER = """
<div style="text-align: center; margin-bottom: 40px;">

# 🔬 Scientific RAG System

**Retrieval-Augmented Generation for Scientific Papers**

This system allows you to ask questions about scientific papers and receive answers with citations from the source documents.

</div>
"""

# =============================================================================
# LLM Provider Configuration
# =============================================================================

LLM_PROVIDERS = {
    "Groq": {
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "default": "llama-3.1-8b-instant",
    },
    "OpenRouter": {
        "models": [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4o-mini",
            "anthropic/claude-3-haiku",
            "meta-llama/llama-3.1-8b-instruct:free",
            "google/gemma-2-9b-it:free",
        ],
        "default": "meta-llama/llama-3.1-8b-instruct:free",
    },
    "OpenAI": {
        "models": [
            "gpt-3.5-turbo",
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
        ],
        "default": "gpt-4o-mini",
    },
}


# =============================================================================
# Mock RAG Pipeline (to be replaced with real implementation)
# =============================================================================


class MockRAGPipeline:
    """
    Mock RAG pipeline for UI development.
    Replace with actual RAGPipeline implementation from scientific_rag.application.rag.pipeline
    """

    def __init__(self):
        self.retrieved_chunks: list[PaperChunk] = []

    def process_query(
        self,
        query: str,
        api_key: str,
        provider: str,
        model: str,
        source_filter: str,
        section_filter: str,
        use_self_query: bool,
        use_query_expansion: bool,
        use_bm25: bool,
        use_dense: bool,
        use_reranking: bool,
        top_k: int,
        expansion_count: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Process a query through the RAG pipeline.

        Returns:
            Tuple of (answer_text, retrieved_chunks_info)
        """
        if not query.strip():
            raise ValueError("Please enter a question.")

        if not api_key.strip():
            raise ValueError("Please enter your API key.")

        if not use_bm25 and not use_dense:
            raise ValueError("Please enable at least one retrieval method (BM25 or Dense).")

        filters = QueryFilters(
            source=source_filter.lower() if source_filter != "Any" else "any",
            section=section_filter.lower() if section_filter != "Any" else "any",
        )

        query_obj = Query(
            text=query,
            filters=filters if (source_filter != "Any" or section_filter != "Any") else None,
            top_k=top_k,
        )

        pipeline_info = []
        if use_self_query:
            pipeline_info.append("✓ Self-Query: Extracting metadata filters")
        if use_query_expansion:
            pipeline_info.append(f"✓ Query Expansion: Generating {expansion_count} variations")
        if use_bm25:
            pipeline_info.append("✓ BM25: Keyword-based retrieval")
        if use_dense:
            pipeline_info.append("✓ Dense: Semantic vector search")
        if use_reranking:
            pipeline_info.append("✓ Reranking: Cross-encoder scoring")

        mock_chunks = self._generate_mock_chunks(query, filters, top_k)
        self.retrieved_chunks = mock_chunks

        answer = self._generate_mock_answer(query, mock_chunks, provider, model)

        chunks_info = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                "source": chunk.source.value,
                "section": chunk.section.value,
                "paper_id": chunk.paper_id,
                "score": chunk.score,
            }
            for chunk in mock_chunks
        ]

        return answer, chunks_info

    def _generate_mock_chunks(self, query: str, filters: QueryFilters, top_k: int) -> list[PaperChunk]:
        mock_chunks = []

        for i in range(min(top_k, 5)):
            source = DataSource.ARXIV if filters.source == "any" or filters.source == "arxiv" else DataSource.PUBMED
            section = (
                SectionType(filters.section)
                if filters.section != "any"
                else [
                    SectionType.INTRODUCTION,
                    SectionType.METHODS,
                    SectionType.RESULTS,
                    SectionType.CONCLUSION,
                ][i % 4]
            )

            chunk = PaperChunk(
                chunk_id=f"mock_chunk_{i}",
                text=f"[Mock content for demonstration] This is a sample text passage from a scientific paper "
                f"that would be relevant to the query: '{query}'. The actual RAG system will retrieve "
                f"real content from the scientific papers dataset based on semantic similarity and "
                f"keyword matching. This chunk is from the {section.value} section of an {source.value} paper.",
                paper_id=f"paper_{1000 + i}",
                source=source,
                section=section,
                position=i,
                score=0.95 - (i * 0.1),
            )
            mock_chunks.append(chunk)

        return mock_chunks

    def _generate_mock_answer(
        self,
        query: str,
        chunks: list[PaperChunk],
        provider: str,
        model: str,
    ) -> str:
        citations = "\n".join(
            [
                f"[{i + 1}] {chunk.text[:100]}... ({chunk.source.value}, {chunk.section.value})"
                for i, chunk in enumerate(chunks[:3])
            ]
        )

        return f"""## Answer

**Note**: This is a demonstration response. The actual RAG system will use {provider}/{model}
to generate answers based on retrieved scientific paper content.

Based on the retrieved documents, here is what the scientific literature says about your question:

"{query}"

The answer would synthesize information from the retrieved chunks, citing specific sources [1], [2], [3].

---

### 📚 Sources

{citations}

---

*Pipeline: {provider} / {model}*
"""


rag_pipeline = MockRAGPipeline()


# =============================================================================
# UI Event Handlers
# =============================================================================


def update_models(provider: str) -> gr.Dropdown:
    if provider in LLM_PROVIDERS:
        models = LLM_PROVIDERS[provider]["models"]
        default = LLM_PROVIDERS[provider]["default"]
        return gr.Dropdown(choices=models, value=default, interactive=True)
    return gr.Dropdown(choices=[], value=None, interactive=True)


def process_query(
    query: str,
    api_key: str,
    provider: str,
    model: str,
    source_filter: str,
    section_filter: str,
    use_self_query: bool,
    use_query_expansion: bool,
    use_bm25: bool,
    use_dense: bool,
    use_reranking: bool,
    top_k: int,
    expansion_count: int,
) -> tuple[str, str, gr.update, gr.update]:
    try:
        answer, chunks = rag_pipeline.process_query(
            query=query,
            api_key=api_key,
            provider=provider,
            model=model,
            source_filter=source_filter,
            section_filter=section_filter,
            use_self_query=use_self_query,
            use_query_expansion=use_query_expansion,
            use_bm25=use_bm25,
            use_dense=use_dense,
            use_reranking=use_reranking,
            top_k=top_k,
            expansion_count=expansion_count,
        )

        chunks_display = format_chunks_display(chunks)

        return answer, chunks_display, gr.update(visible=False), gr.update(visible=True)

    except ValueError as e:
        error_msg = f"⚠️ **Input Error**: {str(e)}"
        return error_msg, "", gr.update(visible=False), gr.update(visible=True)
    except Exception as e:
        error_msg = f"❌ **Error**: {str(e)}"
        return error_msg, "", gr.update(visible=False), gr.update(visible=True)


def format_chunks_display(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No chunks retrieved."

    lines = ["## 📄 Retrieved Chunks\n"]

    for i, chunk in enumerate(chunks, 1):
        score_display = f"{chunk['score']:.3f}" if chunk.get("score") else "N/A"
        lines.append(f"""
### Chunk {i} (Score: {score_display})

**Source**: {chunk["source"].upper()} | **Section**: {chunk["section"].capitalize()} | **Paper ID**: {chunk["paper_id"]}

> {chunk["text"]}

---
""")

    return "\n".join(lines)


# =============================================================================
# Gradio Interface
# =============================================================================


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Scientific RAG System") as demo:
        gr.Markdown(MAIN_HEADER)
        gr.Markdown("---")

        gr.Markdown("## ⚙️ Configuration")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔑 API Settings")

                api_key = gr.Textbox(
                    label="API Key",
                    placeholder="Enter your API key here...",
                    type="password",
                    info="Your API key is not stored and only used for this session",
                )

                provider = gr.Dropdown(
                    label="LLM Provider",
                    choices=list(LLM_PROVIDERS.keys()),
                    value="Groq",
                    info="Select your LLM provider",
                )

                model = gr.Dropdown(
                    label="Model",
                    choices=LLM_PROVIDERS["Groq"]["models"],
                    value=LLM_PROVIDERS["Groq"]["default"],
                    info="Select the model to use",
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Metadata Filters (Optional)")

                source_filter = gr.Dropdown(
                    label="Source",
                    choices=["Any", "ArXiv", "PubMed"],
                    value="Any",
                    info="Filter by paper source",
                )

                section_filter = gr.Dropdown(
                    label="Section",
                    choices=["Any", "Introduction", "Methods", "Results", "Conclusion"],
                    value="Any",
                    info="Filter by paper section",
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Retrieval Parameters")

                top_k = gr.Slider(
                    label="Top-K Results",
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    info="Number of chunks to retrieve",
                )

                expansion_count = gr.Slider(
                    label="Query Expansion Count",
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    info="Number of query variations to generate",
                )

        gr.Markdown("---")

        gr.Markdown("### 🔧 Pipeline Components")
        gr.Markdown("Enable or disable components to customize the retrieval pipeline:")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row(elem_classes="pipeline-component"):
                    with gr.Column(scale=4):
                        gr.Markdown("""
**Self-Query**
Automatically extracts metadata filters from your question
""")
                    with gr.Column(scale=1):
                        use_self_query = gr.Checkbox(
                            label="Enable",
                            value=True,
                            container=False,
                        )

                with gr.Row(elem_classes="pipeline-component"):
                    with gr.Column(scale=4):
                        gr.Markdown("""
**Query Expansion**
Generates variations of your question to improve recall
""")
                    with gr.Column(scale=1):
                        use_query_expansion = gr.Checkbox(
                            label="Enable",
                            value=True,
                            container=False,
                        )

                with gr.Row(elem_classes="pipeline-component"):
                    with gr.Column(scale=4):
                        gr.Markdown("""
**BM25 (Keyword Search)**
Keyword-based retrieval, good for exact terms and rare words
""")
                    with gr.Column(scale=1):
                        use_bm25 = gr.Checkbox(
                            label="Enable",
                            value=True,
                            container=False,
                        )

            with gr.Column(scale=1):
                with gr.Row(elem_classes="pipeline-component"):
                    with gr.Column(scale=4):
                        gr.Markdown("""
**Dense Retrieval (Semantic Search)**
Vector-based retrieval using embeddings, good for semantic meaning
""")
                    with gr.Column(scale=1):
                        use_dense = gr.Checkbox(
                            label="Enable",
                            value=True,
                            container=False,
                        )

                with gr.Row(elem_classes="pipeline-component"):
                    with gr.Column(scale=4):
                        gr.Markdown("""
**Reranking**
Cross-encoder model to improve result relevance
""")
                    with gr.Column(scale=1):
                        use_reranking = gr.Checkbox(
                            label="Enable",
                            value=True,
                            container=False,
                        )

        gr.Markdown("---")

        gr.Markdown("## 💬 Ask a Question")

        query = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What methods are used for protein structure prediction?",
            lines=3,
            info="Enter your question about scientific papers",
        )

        submit_btn = gr.Button(
            "🔍 Search & Generate Answer",
            variant="primary",
            size="lg",
        )

        clear_btn = gr.Button(
            "🗑️ Clear",
            variant="secondary",
        )

        with gr.Group(visible=True) as examples_section:
            gr.Examples(
                examples=[
                    ["What methods are used for protein structure prediction?"],
                    ["How do researchers measure quantum entanglement?"],
                    ["What are the main findings about CRISPR gene editing?"],
                    ["Explain the methodology for clinical trials in cancer treatment"],
                    ["What machine learning techniques are used in medical imaging?"],
                ],
                inputs=[query],
                label="📝 Example Questions",
            )

        with gr.Group(visible=False) as answer_section:
            gr.Markdown("## 📝 Answer")

            answer_output = gr.Markdown(
                label="Generated Answer",
                value="",
            )

            with gr.Accordion("📄 Retrieved Chunks", open=False):
                chunks_output = gr.Markdown(
                    label="Retrieved Chunks",
                    value="",
                )

        provider.change(
            fn=update_models,
            inputs=[provider],
            outputs=[model],
        )

        submit_btn.click(
            fn=process_query,
            inputs=[
                query,
                api_key,
                provider,
                model,
                source_filter,
                section_filter,
                use_self_query,
                use_query_expansion,
                use_bm25,
                use_dense,
                use_reranking,
                top_k,
                expansion_count,
            ],
            outputs=[answer_output, chunks_output, examples_section, answer_section],
        )

        clear_btn.click(
            fn=lambda: ("", "", "", gr.update(visible=True), gr.update(visible=False)),
            inputs=[],
            outputs=[query, answer_output, chunks_output, examples_section, answer_section],
        )

    return demo


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .config-section { border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin: 10px 0; }
        .output-section { min-height: 300px; }
        .pipeline-component { padding: 10px; border-bottom: 1px solid #f0f0f0; }
        footer { display: none !important; }
        """,
    )


if __name__ == "__main__":
    main()
