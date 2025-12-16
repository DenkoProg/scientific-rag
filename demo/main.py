import os
from pathlib import Path
import sys
from typing import Any

import gradio as gr
from loguru import logger


# Auto-configure for HF Spaces
if os.getenv("SPACE_ID"):  # Detect HF Spaces environment
    os.environ.setdefault("QDRANT_URL", ":memory:")
    logger.info("🚀 Running on Hugging Face Spaces with in-memory Qdrant")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

        metadata_badges = []
        if response.generated_query_variations:
            metadata_badges.append(
                f'<span class="metadata-badge">🔍 Query Variations: {len(response.generated_query_variations) + 1}</span>'
            )
        metadata_badges.append(
            f'<span class="metadata-badge">📄 Retrieved Chunks: {len(response.retrieved_chunks)}</span>'
        )
        metadata_badges.append(
            f'<span class="metadata-badge">📊 Display Chunks: {min(display_chunks, len(response.retrieved_chunks))}</span>'
        )
        metadata_badges.append(f'<span class="metadata-badge">⏱️ Execution Time: {response.execution_time:.2f}s</span>')
        metadata_badges.append(f'<span class="metadata-badge">🤖 Model: {provider} / {model}</span>')

        if response.used_filters:
            filters_str = ", ".join([f"{k}={v}" for k, v in response.used_filters.items() if v != "any"])
            if filters_str:
                metadata_badges.append(f'<span class="metadata-badge">🔎 Filters: {filters_str}</span>')

        lines.append('<div class="metadata-container">' + " ".join(metadata_badges) + "</div>")

        return "\n".join(lines)


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
) -> tuple[str, str, gr.update, gr.update, gr.update]:
    try:
        if not rag_pipeline:
            error_msg = "⚠️ **System Error**: RAG Pipeline not initialized. Please check logs."
            return error_msg, "", gr.update(visible=False), gr.update(visible=True), gr.update(value="", visible=False)

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
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value="", visible=False),
        )

    except ValueError as e:
        error_msg = f"⚠️ **Input Error**: {str(e)}"
        return (
            error_msg,
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value="", visible=False),
        )
    except Exception as e:
        error_msg = f"❌ **Error**: {str(e)}"
        return (
            error_msg,
            gr.update(value=""),
            gr.update(visible=False),
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

        gr.Markdown("---")

        gr.Markdown("## ⚙️ Configuration")

        with gr.Row():
            with gr.Column(scale=1):
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
                    value="OpenRouter",
                    info="Select your LLM provider",
                )

                model = gr.Dropdown(
                    label="Model",
                    choices=LLM_PROVIDERS["OpenRouter"]["models"],
                    value=LLM_PROVIDERS["OpenRouter"]["default"],
                    info="Select the model to use",
                )

            with gr.Column(scale=1):
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
            placeholder="e.g., What are quantum error correction approaches?",
            lines=3,
            info="Enter your question about scientific papers",
            elem_id="question_box",
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

        with gr.Row():
            loading_status = gr.Markdown(value="", visible=False, elem_classes="loading-indicator")

        with gr.Group(visible=True, elem_classes="examples-section") as examples_section:
            gr.Markdown("## 📝 Example Questions")

            gr.HTML("""
                <div class="example-questions-container">
                    <button class="example-question-badge" onclick="const ta=document.querySelector('#question_box textarea');if(ta){ta.value='What are quantum error correction approaches?';ta.dispatchEvent(new Event('input',{bubbles:true}));}">What are quantum error correction approaches?</button>
                    <button class="example-question-badge" onclick="const ta=document.querySelector('#question_box textarea');if(ta){ta.value='What is plasma confinement in tokamaks?';ta.dispatchEvent(new Event('input',{bubbles:true}));}">What is plasma confinement in tokamaks?</button>
                    <button class="example-question-badge" onclick="const ta=document.querySelector('#question_box textarea');if(ta){ta.value='When is DNA denaturation?';ta.dispatchEvent(new Event('input',{bubbles:true}));}">When is DNA denaturation?</button>
                </div>
            """)

        with gr.Group(visible=False, elem_classes="answer-section") as answer_section:
            gr.Markdown("## 💡 Your Answer")

            answer_output = gr.Markdown(
                label="Answer",
                value="",
                show_label=False,
                elem_classes="answer-content",
            )

            gr.Markdown("### 📚 Retrieved Chunks")
            chunks_output = gr.Markdown(
                label="Chunks",
                value="",
                show_label=False,
                elem_id="chunks-display",
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
                gr.update(visible=False),
                gr.update(
                    value="<div style='text-align: center; padding: 50px;'><h2 style='color: #1f2937; font-size: 22px; margin-bottom: 12px; font-weight: 600;'>Processing your query...</h2><p style='font-size: 15px; color: #4b5563;'>Retrieving relevant papers and generating answer</p></div>",
                    visible=True,
                ),
            ),
            inputs=[],
            outputs=[answer_output, chunks_output, examples_section, answer_section, loading_status],
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
            outputs=[answer_output, chunks_output, examples_section, answer_section, loading_status],
        )

        clear_btn.click(
            fn=lambda: (
                "",
                "",
                "",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value="", visible=False),
            ),
            inputs=[],
            outputs=[query, answer_output, chunks_output, examples_section, answer_section, loading_status],
        )

    return demo


def main():
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate").set(
            border_color_primary="#ffffff",
        ),
        css="""
        .gradio-container, .contain, body, .gr-box, .gr-form, .gr-panel, .gr-group, .gr-block, .gr-row, .gr-column {
            background: #ffffff !important;
        }

        .main-header { text-align: center; margin-bottom: 20px; }
        .config-section { border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin: 10px 0; }
        .output-section { min-height: 300px; }
        .pipeline-component { padding: 10px; border-bottom: 1px solid #f0f0f0; }

        .examples-section {
            background-color: transparent !important;
            background: #ffffff !important;
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
            padding: 24px !important;
            margin: 20px 0 !important;
        }

        .examples-section h2 {
            font-size: 20px !important;
            font-weight: 600 !important;
            color: #1f2937 !important;
            margin-bottom: 20px !important;
            background: transparent !important;
        }

        .example-questions-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
            background: transparent !important;
            padding: 0 !important;
        }

        .example-question-badge {
            display: inline-block;
            background: #dbeafe;
            color: #1e40af;
            padding: 8px 14px;
            border-radius: 16px;
            font-size: 14px;
            font-weight: 500;
            border: 1px solid #93c5fd;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .example-question-badge:hover {
            background: #bfdbfe;
            border-color: #60a5fa;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
        }

        .loading-indicator {
            background: #ffffff;
            border: 2px solid #667eea;
            color: #1f2937;
            padding: 50px 20px;
            border-radius: 8px;
            text-align: center;
            font-size: 16px;
            font-weight: 500;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            min-height: 200px;
        }

        .answer-container {
            margin-top: 20px;
        }

        .answer-content {
            background: #ffffff;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 20px;
            font-size: 15px;
            line-height: 1.7;
            color: #1f2937;
            min-height: 150px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }

        .chunks-accordion {
            background: #ffffff !important;
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
            margin-top: 16px !important;
        }

        .metadata-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #e5e7eb;
        }

        .metadata-badge {
            display: inline-block;
            background: #dbeafe;
            color: #1e40af;
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 13px;
            font-weight: 500;
            border: 1px solid #93c5fd;
        }

        footer { display: none !important; }
        """,
    )


if __name__ == "__main__":
    main()
