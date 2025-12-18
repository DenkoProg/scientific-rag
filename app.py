from pathlib import Path
import sys
from typing import Any

import gradio as gr
from loguru import logger


sys.path.insert(0, str(Path(__file__).parent / "src"))

from scientific_rag.application.rag.pipeline import RAGPipeline


MAIN_HEADER = """
<div style="text-align: center; margin-bottom: 40px;">

# 🔬 Scientific RAG System

**Retrieval-Augmented Generation for Scientific Papers**

This system allows you to ask questions about scientific papers and receive answers with citations from the source documents.

</div>
"""

LLM_PROVIDERS = {
    "OpenRouter": {
        "models": [
            "meta-llama/llama-3.3-70b-instruct:free",
            "amazon/nova-2-lite-v1:free",
            "qwen/qwen3-235b-a22b:free",
            "openai/gpt-oss-120b:free",
        ],
        "default": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "Groq": {
        "models": [
            "groq/llama-3.1-8b-instant",
            "groq/openai/gpt-oss-120b",
            "groq/qwen/qwen3-32b",
        ],
        "default": "groq/llama-3.1-8b-instant",
    },
}


class RAGPipelineWrapper:
    """Wrapper for RAG pipeline with UI-specific logic."""

    def __init__(self):
        self.pipeline = RAGPipeline()
        logger.info("RAG Pipeline initialized successfully")

    def process_query(
        self,
        query: str,
        api_key: str,
        provider: str,
        model: str,
        use_self_query: bool,
        use_query_expansion: bool,
        use_bm25: bool,
        use_dense: bool,
        use_reranking: bool,
        top_k: int,
        expansion_count: int,
        display_chunks: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Process a query through the RAG pipeline.

        Returns:
            Tuple of (answer_text, retrieved_chunks_info)
        """
        if not query.strip():
            raise ValueError("Please enter a question.")

        if not api_key.strip():
            raise ValueError("Please enter your API key.")

        if not use_bm25 and not use_dense:
            raise ValueError("Please enable at least one retrieval method (BM25 or Dense).")

        if top_k < 1 or top_k > 50:
            raise ValueError("Top-K must be between 1 and 20.")

        if expansion_count < 1 or expansion_count > 5:
            raise ValueError("Query expansion count must be between 1 and 5.")

        if not self.pipeline:
            raise ValueError(
                "RAG Pipeline not initialized. Please run 'make chunk-data' and 'make index-qdrant' first."
            )

        if provider not in LLM_PROVIDERS:
            raise ValueError(f"Invalid provider: {provider}. Must be one of {list(LLM_PROVIDERS.keys())}")

        if model not in LLM_PROVIDERS[provider]["models"]:
            raise ValueError(
                f"Invalid model '{model}' for provider '{provider}'. "
                f"Available models: {LLM_PROVIDERS[provider]['models']}"
            )

        from scientific_rag.application.rag.llm_client import llm_client

        llm_client.api_key = api_key
        llm_client.provider = provider.lower()
        llm_client.model = model

        self.pipeline.query_processor.expand_to_n = expansion_count

        response = self.pipeline.run(
            query=query,
            use_self_query=use_self_query,
            use_query_expansion=use_query_expansion,
            use_bm25=use_bm25,
            use_dense=use_dense,
            use_reranking=use_reranking,
            retrieval_top_k=top_k,
            rerank_top_k=display_chunks,
        )

        answer = self._format_answer(response, provider, model, display_chunks)

        chunks_info = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source.value,
                "section": chunk.section.value,
                "paper_id": chunk.paper_id,
                "score": chunk.score,
            }
            for chunk in response.retrieved_chunks[:display_chunks]
        ]

        return answer, chunks_info

    def _format_answer(self, response, provider: str, model: str, display_chunks: int) -> str:
        """Format RAG response as markdown."""
        lines = []
        lines.append(response.answer)
        lines.append("\n\n")
        lines.append("---\n")
        lines.append("**Metadata:**\n\n")

        if response.generated_query_variations:
            lines.append(f"- 🔍 Query Variations: {len(response.generated_query_variations) + 1}\n")
        lines.append(f"- 📄 Retrieved Chunks: {len(response.retrieved_chunks)}\n")
        lines.append(f"- 📊 Display Chunks: {min(display_chunks, len(response.retrieved_chunks))}\n")
        lines.append(f"- ⏱️ Execution Time: {response.execution_time:.2f}s\n")
        lines.append(f"- 🤖 Model: {provider} / {model}\n")

        if response.used_filters:
            filters_str = ", ".join([f"{k}={v}" for k, v in response.used_filters.items() if v != "any"])
            if filters_str:
                lines.append(f"- 🔎 Filters: {filters_str}\n")

        return "".join(lines)


# Initialize RAG pipeline
try:
    rag_pipeline = RAGPipelineWrapper()
except Exception as e:
    logger.error(f"Failed to initialize RAG pipeline: {e}")
    rag_pipeline = None


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
    use_self_query: bool,
    use_query_expansion: bool,
    use_bm25: bool,
    use_dense: bool,
    use_reranking: bool,
    top_k: int,
    expansion_count: int,
    display_chunks: int,
) -> tuple[str, str, gr.update, gr.update]:
    try:
        if not rag_pipeline:
            error_msg = "⚠️ **System Error**: RAG Pipeline not initialized. Please check logs."
            return error_msg, "", gr.update(visible=True), gr.update(value="", visible=False)

        answer, chunks = rag_pipeline.process_query(
            query=query,
            api_key=api_key,
            provider=provider,
            model=model,
            use_self_query=use_self_query,
            use_query_expansion=use_query_expansion,
            use_bm25=use_bm25,
            use_dense=use_dense,
            use_reranking=use_reranking,
            top_k=top_k,
            expansion_count=expansion_count,
            display_chunks=display_chunks,
        )

        chunks_display = format_chunks_display(chunks)

        return (
            answer,
            gr.update(value=chunks_display),
            gr.update(visible=True),
            gr.update(value="", visible=False),
        )

    except ValueError as e:
        error_msg = f"⚠️ **Input Error**: {str(e)}"
        return (
            error_msg,
            gr.update(value=""),
            gr.update(visible=True),
            gr.update(value="", visible=False),
        )
    except Exception as e:
        error_msg = f"❌ **Error**: {str(e)}"
        return (
            error_msg,
            gr.update(value=""),
            gr.update(visible=True),
            gr.update(value="", visible=False),
        )


def format_chunks_display(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No chunks retrieved."

    lines = []

    for i, chunk in enumerate(chunks, 1):
        score_display = f"{chunk['score']:.3f}" if chunk.get("score") else "N/A"
        lines.append(f"""
### Chunk {i} (Score: {score_display})

**Source**: {chunk["source"].upper()} | **Section**: {chunk["section"].capitalize()} | **Paper ID**: {chunk["paper_id"]}

> {chunk["text"]}

---
""")

    return "\n".join(lines)


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Scientific RAG System") as demo:
        gr.Markdown(MAIN_HEADER)

        with gr.Accordion("⚙️ Configuration", open=True):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔑 API Settings")

                    api_key = gr.Textbox(
                        label="API Key",
                        placeholder="Enter your API key here...",
                        type="password",
                        value="",
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

                with gr.Column():
                    gr.Markdown("### 📊 Retrieval Parameters")

                    top_k = gr.Slider(
                        label="Top-K chunks to retrieve",
                        minimum=1,
                        maximum=50,
                        value=20,
                        step=1,
                        info="Initial retrieval before reranking",
                    )

                    display_chunks = gr.Slider(
                        label="Top-K chunks after reranking",
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        info="Final chunks to use for answer generation",
                    )

                    expansion_count = gr.Slider(
                        label="Query expansion count",
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        info="Number of query variations to generate",
                    )

            gr.Markdown("### 🔧 Pipeline Components")

            with gr.Row():
                with gr.Column():
                    use_self_query = gr.Checkbox(
                        label="🔎 Self-Query (Extract filters from question)",
                        value=True,
                        info="Automatically extracts metadata filters from your question",
                    )

                    use_query_expansion = gr.Checkbox(
                        label="🔄 Query Expansion (Generate variations)",
                        value=True,
                        info="Generates variations of your question to improve recall",
                    )

                    use_bm25 = gr.Checkbox(
                        label="🔤 BM25 (Keyword Search)",
                        value=True,
                        info="Keyword-based retrieval, good for exact terms",
                    )

                with gr.Column():
                    use_dense = gr.Checkbox(
                        label="🧠 Dense Retrieval (Semantic Search)",
                        value=True,
                        info="Vector-based retrieval using embeddings",
                    )

                    use_reranking = gr.Checkbox(
                        label="🎯 Reranking", value=True, info="Cross-encoder model to improve result relevance"
                    )

        gr.Markdown("## 💬 Ask a Question")

        query = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What are quantum error correction approaches?",
            lines=3,
            info="Enter your question about scientific papers",
        )

        gr.Examples(
            examples=[
                "What are quantum error correction approaches?",
                "What is plasma confinement in tokamaks?",
                "When is DNA denaturation?",
            ],
            inputs=query,
            label="📝 Example Questions",
        )

        with gr.Row():
            submit_btn = gr.Button(
                "🔍 Search & Generate Answer",
                variant="primary",
                size="lg",
            )

            clear_btn = gr.Button(
                "🗑️ Clear",
                variant="secondary",
            )

        loading_status = gr.Markdown(value="", visible=False)

        with gr.Group(visible=False) as answer_section:
            gr.Markdown("## 💡 Your Answer")

            answer_output = gr.Markdown(
                label="Answer",
                value="",
                show_label=False,
                elem_id="answer-content",
            )

            gr.Markdown("<br>")  # Spacer

            with gr.Accordion("📚 Retrieved Chunks", open=True):
                chunks_output = gr.Markdown(
                    label="Chunks",
                    value="",
                    show_label=False,
                )

        provider.change(
            fn=update_models,
            inputs=[provider],
            outputs=[model],
        )

        submit_btn.click(
            fn=lambda *args: (
                "",
                "",
                gr.update(visible=False),
                gr.update(
                    value="⏳ **Processing your query...** Retrieving relevant papers and generating answer",
                    visible=True,
                ),
            ),
            inputs=[],
            outputs=[answer_output, chunks_output, answer_section, loading_status],
        ).then(
            fn=process_query,
            inputs=[
                query,
                api_key,
                provider,
                model,
                use_self_query,
                use_query_expansion,
                use_bm25,
                use_dense,
                use_reranking,
                top_k,
                expansion_count,
                display_chunks,
            ],
            outputs=[answer_output, chunks_output, answer_section, loading_status],
        )

        clear_btn.click(
            fn=lambda: (
                "",
                "",
                "",
                gr.update(visible=False),
                gr.update(value="", visible=False),
            ),
            inputs=[],
            outputs=[query, answer_output, chunks_output, answer_section, loading_status],
        )

    return demo


css = """
    #answer-content {
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    """


def main():
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, css=css)


if __name__ == "__main__":
    main()
