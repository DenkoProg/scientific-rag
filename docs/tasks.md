# Scientific RAG - Implementation Tasks

> **Project**: Scientific Advanced RAG System
> **Dataset**: [armanc/scientific_papers](https://huggingface.co/datasets/armanc/scientific_papers) (ArXiv + PubMed)
> **Deadline**: December 16, 2025
> **Reference Architecture**: LLM-Engineers-Handbook (Domain-Driven Design)

---

## Overview

Build a Retrieval-Augmented Generation (RAG) system for answering questions about scientific papers. The system will use the `armanc/scientific_papers` dataset containing ~320K papers from ArXiv and PubMed with articles, abstracts, and section names.

### Dataset Structure

```python
{
    "abstract": "Summary of the paper...",
    "article": "Full body of the paper, paragraphs separated by \\n...",
    "section_names": "[sec:introduction]introduction\\n[sec:methods]methods\\n..."
}
```

- **arxiv**: 203,037 train / 6,436 val / 6,440 test
- **pubmed**: 119,924 train / 6,633 val / 6,658 test

---

## Project Structure (Target)

```
scientific-rag/
├── pyproject.toml              # Project configuration
├── Makefile                    # Development commands
├── docker-compose.yaml         # Qdrant infrastructure
├── .env.dist                   # Environment template
├── README.md                   # Documentation
├── app.py                      # Main Gradio application
├── requirements.txt            # HuggingFace Spaces dependencies
├── docs/
│   ├── assignment.md           # Assignment requirements
│   ├── tasks.md                # This file
│   ├── roles.md                # Team roles distribution
│   └── submission.md           # Submission requirements
├── data/
│   ├── raw/                    # Downloaded dataset cache
│   └── processed/              # Processed chunks (JSON)
├── notebooks/                  # Jupyter notebooks for testing
│   ├── 01_test_data_loader.ipynb
│   ├── 02_test_rag_pipeline.ipynb
│   └── 03_chunks_eda.ipynb
├── src/
│   └── scientific_rag/         # Main package (under src/)
│       ├── __init__.py
│       ├── cli.py              # Typer CLI for data pipeline
│       ├── settings.py         # Pydantic settings management
│       ├── domain/             # Core entities
│       │   ├── __init__.py
│       │   ├── documents.py    # ScientificPaper, PaperChunk models
│       │   ├── queries.py      # Query, QueryFilters, ExpandedQuery models
│       │   ├── responses.py    # RAGResponse model
│       │   └── types.py        # DataSource, SectionType enums
│       ├── application/        # Business logic
│       │   ├── data_loader.py  # HuggingFace dataset loading
│       │   ├── chunking/
│       │   │   └── scientific_chunker.py  # Section-aware chunking
│       │   ├── embeddings/
│       │   │   └── encoder.py  # E5 embedding model wrapper
│       │   ├── query/          # Query processing (unified module)
│       │   │   └── query_processor.py  # Self-query + expansion combined
│       │   ├── retrieval/
│       │   │   ├── bm25_retriever.py    # Sparse search via Qdrant
│       │   │   ├── dense_retriever.py   # Dense search via Qdrant
│       │   │   └── hybrid_retriever.py  # Combined retrieval
│       │   ├── reranking/
│       │   │   └── cross_encoder.py     # Cross-encoder reranker
│       │   └── rag/
│       │       ├── pipeline.py          # Main RAG orchestration
│       │       ├── prompt_templates.py  # System and RAG prompts
│       │       └── llm_client.py        # LiteLLM wrapper
│       ├── infrastructure/
│       │   └── qdrant.py       # Qdrant client (dense + sparse vectors)
│       └── scripts/
│           ├── chunk_data.py   # Data chunking script
│           └── index_qdrant.py # Qdrant indexing script
└── tests/
    └── (test files)
```

---

## Implementation Tasks

### Phase 1: Project Setup & Data Loading

- [✅] **1.1** Update `pyproject.toml` with project dependencies

  - `datasets` - HuggingFace datasets (pinned <3.0.0 for compatibility)
  - `sentence-transformers` - Embeddings and cross-encoders
  - `fastembed` - Sparse BM25 embeddings for Qdrant
  - `qdrant-client` - Vector database client
  - `litellm` - LLM abstraction layer
  - `gradio` - UI framework (pinned 6.1.0)
  - `pydantic` - Data validation
  - `pydantic-settings` - Configuration management
  - `loguru` - Logging
  - `numpy`, `scipy` - Numerical operations
  - `tqdm` - Progress bars
  - `typer` - CLI framework
  - `nltk` - Natural language processing
  - `tenacity` - Retry logic for API calls
  - `rootutils` - Path management

- [✅] **1.2** Create `docker-compose.yaml` for local infrastructure

  - Qdrant vector database service
  - Example:
    ```yaml
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - "6333:6333"
          - "6334:6334"
        volumes:
          - qdrant_storage:/qdrant/storage
    volumes:
      qdrant_storage:
    ```
  - Add `make qdrant-up` and `make qdrant-down` commands

- [✅] **1.3** Create `src/scientific_rag/settings.py`

  - Pydantic BaseSettings with `.env` file support
  - Qdrant settings: `qdrant_url`, `qdrant_api_key`, `qdrant_collection_name`
  - Embedding settings: `embedding_model_name`, `sparse_embedding_model_name`, `embedding_device`
  - Chunking settings: `chunk_size`, `chunk_overlap`, `min_chunk_size`
  - Retrieval settings: `retrieval_top_k`, `bm25_weight`, `dense_weight`
  - Reranking settings: `reranker_model_name`, `rerank_top_k`
  - LLM settings: `llm_provider`, `llm_model`, `llm_api_key`, `llm_temperature`, `llm_max_tokens`
  - Dataset settings: `dataset_name`, `dataset_split`, `dataset_cache_dir`, `dataset_sample_size`

- [✅] **1.4** Create `src/scientific_rag/domain/` entities

  - `types.py`: StrEnum for `DataSource` (ARXIV, PUBMED) and `SectionType` (INTRODUCTION, METHODS, RESULTS, CONCLUSION, ABSTRACT, OTHER)
  - `documents.py`: `ScientificPaper` (frozen model), `PaperChunk` with score/embedding fields and `to_dict()` method
  - `queries.py`: `QueryFilters` with `to_qdrant_filter()`, `Query`, `ExpandedQuery` with `all_queries()`, `EmbeddedQuery`
  - `responses.py`: `RAGResponse` with answer, query variations, chunks, filters, execution_time

- [✅] **1.5** Implement `src/scientific_rag/application/data_loader.py`
  - Load `armanc/scientific_papers` from HuggingFace
  - Support both `arxiv` and `pubmed` subsets
  - Configurable sample size for development via settings
  - Progress tracking with tqdm
  - `load_papers()` and `load_both_sources()` methods

- [✅] **1.6** Create `src/scientific_rag/cli.py` with Typer
  - `chunk` command: Process papers and generate chunks
  - `index` command: Embed chunks and upload to Qdrant
  - `pipeline` command: Run complete pipeline (chunk + index)
  - `info` command: Show configuration and Qdrant status

### Phase 2: Chunking Strategy

- [✅] **2.1** Implement `src/scientific_rag/application/chunking/scientific_chunker.py`

  - **Paragraph-based splitting**: Split on `\n` boundaries, group into chunks
  - **Word-based sizing**: `chunk_size` and `chunk_overlap` in words
  - **Minimum chunk filtering**: Skip chunks below `min_chunk_size`
  - **Abstract handling**: Create separate chunk for abstract with `SectionType.ABSTRACT`
  - **Section inference**: Infer section type by position (intro/methods/results/conclusion)
  - **Metadata preservation**: Store source, section, paper_id, position
  - **LaTeX normalization**: Handle `@xmath`, `@xcite` placeholders
  - **Hash-based IDs**: Generate UUID5 chunk IDs from paper_id + position

- [✅] **2.2** Create `src/scientific_rag/scripts/chunk_data.py`
  - Batch processing with configurable batch_size
  - Stream-write to JSON file (memory efficient)
  - Generate statistics file with chunk counts
  - Output to `data/processed/chunks_{split}.json`

### Phase 3: Retrieval Implementation

- [✅] **3.1** Create `src/scientific_rag/application/embeddings/encoder.py`

  - Singleton pattern for `intfloat/e5-small-v2` model
  - Query/passage prefixing for E5 models (`query: ` / `passage: `)
  - Batch embedding with configurable batch_size
  - Normalized embeddings
  - CPU/GPU device configuration via settings

- [✅] **3.2** Implement `src/scientific_rag/infrastructure/qdrant.py`

  - QdrantService class with sync client
  - Support for local Docker and Qdrant Cloud (via `qdrant_url`, `qdrant_api_key`)
  - **Hybrid vector collection**: Dense vectors (384-d) + Sparse BM25 vectors
  - `create_collection()` with named vectors: "dense" and "bm25"
  - Payload indexes on `source`, `section`, `paper_id` fields
  - `upsert_chunks()` with dense + sparse embeddings
  - `search_dense()` for semantic search
  - `search_sparse()` for BM25 search
  - `_build_filters()` to convert `QueryFilters` to Qdrant Filter objects
  - `get_collection_info()` for status checks

- [✅] **3.3** Implement `src/scientific_rag/application/retrieval/bm25_retriever.py`

  - Use `fastembed.SparseTextEmbedding` with `Qdrant/bm25` model
  - Sparse vector search via Qdrant (not in-memory rank_bm25)
  - Score normalization (min-max scaling)
  - `search(query, k, filters) -> List[PaperChunk]` interface

- [✅] **3.4** Implement `src/scientific_rag/application/retrieval/dense_retriever.py`

  - Semantic search using Qdrant dense vectors
  - Use shared `encoder` singleton for query embedding
  - Apply metadata filters from `QueryFilters`
  - `search(query, k, filters) -> List[PaperChunk]` interface

- [✅] **3.5** Implement `src/scientific_rag/application/retrieval/hybrid_retriever.py`
  - Combine BM25 and dense retrieval with weighted scoring
  - Configurable weights: `bm25_weight`, `dense_weight` from settings
  - Toggle switches: `use_bm25`, `use_dense`
  - Merge results by chunk_id, sum weighted scores
  - Return top-k by combined score

### Phase 4: Query Processing & Metadata Filtering

- [✅] **4.1** Implement `src/scientific_rag/application/query/query_processor.py` (Unified module)

  - **Self-Query (filter extraction)**:
    - Rule-based regex matching for source detection ("arxiv", "pubmed")
    - Section detection patterns: introduction, methods, results, conclusion
    - Keep original query unmodified (no stripping of filter keywords)
    - Return `QueryFilters` object

  - **Query Expansion**:
    - LLM-based query variation generation
    - Configurable `expand_to_n` parameter (default: 3)
    - Generate N-1 variations to get N total queries
    - Parse LLM response by "###" separator
    - Skip expansion if no API key configured

  - **Combined processing**:
    - `process(query, use_expansion, extract_filters) -> ExpandedQuery`
    - Returns original query, variations list, and filters

- [✅] **4.2** Update `src/scientific_rag/domain/queries.py`

  - `QueryFilters` model with `to_qdrant_filter()` method
  - `ExpandedQuery` model with `all_queries()` helper method
  - Example:

    ```python
    class QueryFilters(BaseModel):
        source: Literal["arxiv", "pubmed", "any"] = "any"
        section: Literal["introduction", "methods", "results", "conclusion", "any"] = "any"

        def to_qdrant_filter(self) -> dict | None:
            # Build Qdrant filter syntax

    class ExpandedQuery(BaseModel):
        original: str
        variations: list[str]
        filters: QueryFilters | None = None

        def all_queries(self) -> list[str]:
            return [self.original] + self.variations
    ```

### Phase 5: Reranking

- [✅] **5.1** Implement `src/scientific_rag/application/reranking/cross_encoder.py`
  - Use `cross-encoder/ms-marco-MiniLM-L6-v2`
  - Singleton pattern with lazy loading
  - `rerank(query, chunks, top_k) -> List[PaperChunk]` interface
  - Copy chunks before modifying scores (immutability)
  - Score-based sorting, return top_k

### Phase 6: LLM Integration

- [✅] **6.1** Implement `src/scientific_rag/application/rag/llm_client.py`

  - LiteLLM wrapper for provider abstraction
  - Support for OpenRouter, Groq (model format: `provider/model`)
  - Retry logic with `tenacity` (3 attempts, exponential backoff)
  - Custom headers for OpenRouter (HTTP-Referer, X-Title)
  - Singleton pattern for shared client
  - Dynamic API key/model override from UI

- [✅] **6.2** Create `src/scientific_rag/application/rag/prompt_templates.py`

  - `RAGPrompts` class with static methods
  - `SYSTEM_PROMPT`: Scientific assistant with citation rules
  - `format_context(chunks)`: Format chunks with [1], [2] labels and metadata
  - `generate_rag_prompt(query, chunks)`: Full prompt with context injection
  - Citation format: `[{i}] Source: {source} | Paper ID: {id} | Section: {section}`

- [✅] **6.3** Implement `src/scientific_rag/application/rag/pipeline.py`
  - Main `RAGPipeline` class with dependency injection
  - Initialize: `QueryProcessor`, `HybridRetriever`, `CrossEncoderReranker`, `LLMClient`
  - `run()` method with toggle parameters
  - Full pipeline flow:
    ```
    1. Query Processing: Extract filters + expand query variations
    2. Retrieve: Search with all queries via HybridRetriever
    3. Merge & Deduplicate: Combine by chunk_id, keep highest scores
    4. Rerank: Cross-encoder scoring (optional)
    5. Generate: LLM with RAGPrompts template
    ```
  - Return `RAGResponse` with answer, chunks, execution_time, metadata

### Phase 7: User Interface

- [✅] **7.1** Create `app.py` (root level) with Gradio

  - `RAGPipelineWrapper` class for UI-specific logic
  - Text input for questions
  - Password field for API key (not stored in code)
  - Dropdown for LLM provider (OpenRouter, Groq)
  - Dynamic model dropdown based on selected provider
  - Checkboxes for pipeline components:
    - [✅] Enable Self-Query (metadata extraction)
    - [✅] Enable Query Expansion
    - [✅] Enable BM25
    - [✅] Enable Dense Retrieval
    - [✅] Enable Reranking
  - Sliders:
    - Top-K retrieval (1-50)
    - Query expansion count (1-5)
    - Display chunks count (1-10)
  - Tabbed output:
    - Answer tab: Formatted markdown with metadata
    - Retrieved Chunks tab: JSON display of chunk details
  - Example queries with different scenarios

- [✅] **7.2** Add service description

  - Header with system explanation
  - Model providers and available models

- [✅] **7.3** Style and UX improvements
  - Clean Gradio Blocks layout
  - Input validation with error messages
  - Loading states during processing
  - Clear button functionality

### Phase 8: Deployment

- [✅] **8.1** Create `requirements.txt` for HuggingFace Spaces

  - Pin versions for reproducibility
  - Include all production dependencies

- [✅] **8.2** Create HuggingFace Space configuration

  - `README.md` with YAML frontmatter:
    ```yaml
    ---
    title: Scientific RAG System
    emoji: 🔬
    sdk: gradio
    sdk_version: 6.0.0
    app_file: app.py
    python_version: 3.11
    ---
    ```
  - Configure Qdrant Cloud connection for deployment

- [✅] **8.3** Deploy to HuggingFace Spaces
  - Test with sample queries
  - Verify API key handling
  - Verify Qdrant Cloud connectivity

### Phase 9: Evaluation & Documentation

- [✅] **9.1** Find queries where BM25 outperforms dense retrieval

  - Queries with specific terminology, rare words, or exact phrases
  - Examples:
    - "papers mentioning @xmath0 decay channel"
    - "CLEO detector measurements"

- [✅] **9.2** Find queries where dense retrieval outperforms BM25

  - Semantic similarity queries
  - Paraphrased questions
  - Examples:
    - "How do researchers measure particle lifetimes?"
    - "What methods are used for blood clot prevention?"

- [✅] **9.3** Demonstrate metadata filtering effectiveness

  - Show queries where filtering by source improves results
  - Show queries where filtering by section improves results
  - Examples:
    - "arxiv papers about quantum computing" → filter to arxiv
    - "methodology for clinical trials" → filter to methods section

- [✅] **9.4** Document the system in README.md

  - Architecture diagram
  - Installation instructions (UV, Docker, Qdrant)
  - Quick start guide with Make commands
  - CLI usage documentation
  - Environment configuration

- [✅] **9.5** Prepare submission materials
  - Source code link (GitHub)
  - Deployed service link (HuggingFace Spaces)
  - Component checklist (per assignment requirements)

---

## Optional Enhancements (Bonus Points)

### Citation Enhancement

- [ ] **B.1** Improve citation formatting
  - Parse and display chunk source information
  - Show paper abstract or section name
  - Link citations to source documents

### Performance Optimization

- [ ] **B.2** Add caching layer

  - Cache embeddings
  - Cache LLM responses for identical queries

- [ ] **B.3** Optimize for larger dataset
  - FAISS index for fast similarity search
  - Batch processing improvements

---

## Dependencies Summary

```toml
[project]
name = "scientific-rag"
version = "0.1.0"
description = "Scientific Papers RAG System with Advanced Retrieval"
requires-python = ">=3.11"

dependencies = [
    # Data
    "datasets<3.0.0",
    "huggingface-hub>=0.20.0",

    # ML/Embeddings
    "sentence-transformers>=3.0.0",
    "torch>=2.0.0",
    "numpy>=1.26.0",
    "scipy>=1.11.0",

    # Retrieval
    "fastembed>=0.2.0",  # For BM25 sparse embeddings
    "qdrant-client>=1.8.0",

    # LLM
    "litellm>=1.0.0",
    "tenacity>=9.1.2",  # Retry logic

    # Configuration
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",

    # UI
    "gradio==6.1.0",

    # CLI
    "typer>=0.9.0",

    # Utilities
    "loguru>=0.7.0",
    "tqdm>=4.65.0",
    "python-dotenv>=1.0.0",
    "rootutils",
    "nltk>=3.9.2",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
    "pre-commit>=3.0.0",
    "ipykernel>=6.0.0",
    "ipywidgets",
]

[project.scripts]
cli = "scientific_rag.cli:app"
```

---

## Quick Start Commands

```bash
# Setup
make install

# Start Qdrant
make qdrant-up

# Process data pipeline
make pipeline        # Complete: chunk + index
make chunk-data      # Just chunking
make index-qdrant    # Just indexing

# Run application
make run-app

# Development
make lint
make format
make clean

# CLI commands
uv run cli chunk --batch-size 10000
uv run cli index --embedding-batch-size 32
uv run cli pipeline
uv run cli info
```

---

## Key Implementation Notes

### Chunking Strategy

For scientific papers:

1. **Paragraph-based chunking**: Split on `\n` boundaries, group by word count
2. **Abstract handling**: Create dedicated chunk for abstract with `SectionType.ABSTRACT`
3. **Section inference**: Infer section type by position in document (first 15% = intro, etc.)
4. **Handle LaTeX**: Normalize `@xmath`, `@xcite` placeholders with readable markers
5. **Hash-based IDs**: UUID5 from paper_id + position for deterministic chunk IDs

### BM25 Implementation

Instead of in-memory `rank_bm25`:

- Use `fastembed.SparseTextEmbedding` with `Qdrant/bm25` model
- Store sparse vectors directly in Qdrant alongside dense vectors
- Enables native filtering on BM25 search (not possible with in-memory BM25)

### Hybrid Search

Both BM25 and dense search support:

- Qdrant native filtering by `source` and `section`
- Weighted score combination (configurable weights)
- Unified result merging by chunk_id

### Retrieval Comparison

Document specific queries that demonstrate:

- BM25 strength: Exact term matching, rare terminology, specific identifiers
- Dense strength: Semantic understanding, paraphrased queries, conceptual similarity

### LLM Configuration

Supported providers:

- **OpenRouter**: Multiple free models including `meta-llama/llama-3.3-70b-instruct:free`
- **Groq**: Fast inference with `groq/llama-3.1-8b-instant`

### Citation Format

```
Answer: The decay channel measurement shows... [1]. Further analysis using the CLEO detector... [2].

[1] Source: ARXIV | Paper ID: arxiv_123 | Section: introduction
    Content: "we have studied the leptonic decay..."

[2] Source: ARXIV | Paper ID: arxiv_456 | Section: methods
    Content: "data collected with the CLEO detector..."
```

---

## Timeline Suggestion

| Days               | Focus Area                                |
| ------------------ | ----------------------------------------- |
| Day 1-2 (Dec 12-13)| Phase 1-2: Setup, Data Loading, Chunking  |
| Day 2-3 (Dec 13-14)| Phase 3-4: Retrieval, Query Processing    |
| Day 3-4 (Dec 14-15)| Phase 5-6: Reranking, LLM Integration     |
| Day 4-5 (Dec 15-16)| Phase 7-9: UI, Deployment, Documentation  |

---

## References

- [Assignment Document](./docs/assignment.md)
- [LLM-Engineers-Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook) - Reference architecture
- [Scientific Papers Dataset](https://huggingface.co/datasets/armanc/scientific_papers)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Gradio Documentation](https://www.gradio.app/docs)
