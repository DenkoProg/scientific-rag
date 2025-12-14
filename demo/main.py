import json
from pathlib import Path
import sys
from typing import Any

import gradio as gr
from loguru import logger


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scientific_rag.application.rag.pipeline import RAGPipeline
from scientific_rag.domain.documents import PaperChunk
from scientific_rag.domain.queries import Query, QueryFilters
from scientific_rag.domain.types import DataSource, SectionType
from scientific_rag.settings import settings


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
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-2-9b-it:free",
        ],
        "default": "meta-llama/llama-3.3-70b-instruct:free",
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
# RAG Pipeline Initialization
# =============================================================================


def load_chunks() -> list[PaperChunk]:
    """Load chunks from processed data file."""
    chunks_file = Path(settings.root_dir) / "data" / "processed" / f"chunks_{settings.dataset_split}.json"

    if not chunks_file.exists():
        logger.warning(f"Chunks file not found: {chunks_file}")
        logger.warning("Please run 'make chunk-data' to generate chunks first.")
        return []

    logger.info(f"Loading chunks from {chunks_file}")
    with open(chunks_file, encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = [PaperChunk(**chunk_data) for chunk_data in chunks_data]
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


class RAGPipelineWrapper:
    """Wrapper for RAG pipeline with UI-specific logic."""

    def __init__(self):
        self.chunks = load_chunks()
        if self.chunks:
            self.pipeline = RAGPipeline(self.chunks)
            logger.info("RAG Pipeline initialized successfully")
        else:
            self.pipeline = None
            logger.warning("RAG Pipeline not initialized - no chunks available")

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
        # Validation
        if not query.strip():
            raise ValueError("Please enter a question.")

        if not api_key.strip():
            raise ValueError("Please enter your API key.")

        if not use_bm25 and not use_dense:
            raise ValueError("Please enable at least one retrieval method (BM25 or Dense).")

        if not self.pipeline:
            raise ValueError(
                "RAG Pipeline not initialized. Please run 'make chunk-data' and 'make index-qdrant' first."
            )

        # Set LLM API key dynamically
        from scientific_rag.application.rag.llm_client import llm_client

        # Map provider names to LiteLLM format
        provider_map = {
            "Groq": "groq",
            "OpenRouter": "openrouter",
            "OpenAI": "openai",
        }

        llm_provider = provider_map.get(provider, provider.lower())
        llm_client.api_key = api_key
        llm_client.provider = llm_provider
        llm_client.model = model

        # Update query processor expansion count
        self.pipeline.query_processor.expand_to_n = expansion_count

        # Run pipeline
        response = self.pipeline.run(
            query=query,
            use_self_query=use_self_query,
            use_query_expansion=use_query_expansion,
            use_bm25=use_bm25,
            use_dense=use_dense,
            use_reranking=use_reranking,
            retrieval_top_k=top_k,
            rerank_top_k=min(top_k, 5),  # Rerank top 5 or less
        )

        # Format answer with metadata
        answer = self._format_answer(response, provider, model)

        # Format chunks for display
        chunks_info = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                "source": chunk.source.value,
                "section": chunk.section.value,
                "paper_id": chunk.paper_id,
                "score": chunk.score,
            }
            for chunk in response.retrieved_chunks
        ]

        return answer, chunks_info

    def _format_answer(self, response, provider: str, model: str) -> str:
        """Format RAG response as markdown."""
        lines = ["## Answer\n"]
        lines.append(response.answer)
        lines.append("\n---\n")

        # Add metadata
        metadata_parts = []
        if response.generated_query_variations:
            metadata_parts.append(f"🔍 **Query Variations**: {len(response.generated_query_variations) + 1}")
        metadata_parts.append(f"📄 **Retrieved Chunks**: {len(response.retrieved_chunks)}")
        metadata_parts.append(f"⏱️ **Execution Time**: {response.execution_time:.2f}s")
        metadata_parts.append(f"🤖 **Model**: {provider} / {model}")

        if response.used_filters:
            filters_str = ", ".join([f"{k}={v}" for k, v in response.used_filters.items() if v != "any"])
            if filters_str:
                metadata_parts.append(f"🔎 **Filters**: {filters_str}")

        lines.append(" | ".join(metadata_parts))

        return "\n".join(lines)


# Initialize RAG pipeline
try:
    rag_pipeline = RAGPipelineWrapper()
except Exception as e:
    logger.error(f"Failed to initialize RAG pipeline: {e}")
    rag_pipeline = None


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

        # System status
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
