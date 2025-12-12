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
├── tasks.md                    # This file
├── docs/
│   └── assignment.md           # Assignment requirements
├── configs/
│   └── rag_config.yaml         # RAG pipeline configuration
├── data/
│   ├── raw/                    # Downloaded dataset cache
│   └── processed/              # Processed chunks
├── scientific_rag/             # Main package
│   ├── __init__.py
│   ├── settings.py             # Configuration management
│   ├── domain/                 # Core entities
│   │   ├── __init__.py
│   │   ├── documents.py        # Document models (Paper, Chunk)
│   │   ├── queries.py          # Query models
│   │   └── types.py            # Enums and type definitions
│   ├── application/            # Business logic
│   │   ├── __init__.py
│   │   ├── data_loader.py      # HuggingFace dataset loading
│   │   ├── chunking/           # Chunking strategies
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # Abstract chunker
│   │   │   └── scientific_chunker.py
│   │   ├── embeddings/         # Embedding models
│   │   │   ├── __init__.py
│   │   │   └── encoder.py      # Sentence-transformers wrapper
│   │   ├── query_processing/   # Query enhancement
│   │   │   ├── __init__.py
│   │   │   ├── query_expansion.py   # Multi-query generation
│   │   │   └── self_query.py        # Metadata extraction
│   │   ├── retrieval/          # Retrieval logic
│   │   │   ├── __init__.py
│   │   │   ├── bm25_retriever.py
│   │   │   ├── dense_retriever.py
│   │   │   └── hybrid_retriever.py
│   │   ├── reranking/          # Reranker
│   │   │   ├── __init__.py
│   │   │   └── cross_encoder.py
│   │   └── rag/                # RAG pipeline
│   │       ├── __init__.py
│   │       ├── pipeline.py     # Main RAG orchestration
│   │       ├── prompt_templates.py
│   │       └── llm_client.py   # LiteLLM wrapper
│   └── infrastructure/         # External integrations
│       ├── __init__.py
│       └── qdrant.py           # Qdrant vector database client
├── demo/                        # Gradio/Streamlit UI
│   ├── __init__.py
│   └── main.py                 # Web interface
└── tests/
    ├── __init__.py
    ├── unit/
    │   ├── test_chunking.py
    │   ├── test_retrieval.py
    │   └── test_reranking.py
    └── integration/
        └── test_rag_pipeline.py
```

---

## Implementation Tasks

### Phase 1: Project Setup & Data Loading

- [✅] **1.1** Update `pyproject.toml` with project dependencies

  - `datasets` - HuggingFace datasets
  - `sentence-transformers` - Embeddings and cross-encoders
  - `rank-bm25` - BM25 retrieval
  - `qdrant-client` - Vector database client
  - `litellm` - LLM abstraction layer
  - `gradio` or `streamlit` - UI framework
  - `pydantic` - Data validation
  - `pydantic-settings` - Configuration management
  - `loguru` - Logging
  - `numpy`, `scipy` - Numerical operations
  - `tqdm` - Progress bars

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

- [✅] **1.3** Create `scientific_rag/settings.py`

  - Environment variable management
  - Model IDs configuration
  - API keys handling (OpenAI, Groq, OpenRouter)
  - Qdrant connection settings (host, port, API key for cloud)
  - Default chunking parameters

- [✅] **1.4** Create `scientific_rag/domain/` entities

  - `types.py`: Enums for DataSource (ARXIV, PUBMED), SectionType
  - `documents.py`: `ScientificPaper`, `PaperChunk` Pydantic models with metadata
  - `queries.py`: `Query`, `EmbeddedQuery`, `QueryFilters` models

- [✅] **1.5** Implement `scientific_rag/application/data_loader.py`
  - Load `armanc/scientific_papers` from HuggingFace
  - Support both `arxiv` and `pubmed` subsets
  - Configurable sample size for development
  - Progress tracking with tqdm

### Phase 2: Chunking Strategy

- Configurable sample size for development
- Progress tracking with tqdm

### Phase 2: Chunking Strategy

- [✅] **2.1** Implement `scientific_rag/application/chunking/scientific_chunker.py`

  - **Section-aware chunking**: Parse `section_names` to identify sections
  - **Paragraph-based splitting**: Split on `\n` boundaries
  - **Overlap strategy**: Add overlap between chunks for context
  - Configurable `chunk_size` and `chunk_overlap`
  - **Metadata preservation**: Store source (arxiv/pubmed), normalized section name, paper_id, position
  - Normalize section names to enum values (introduction, methods, results, conclusion, other)

- [✅] **2.2** Create processing script to generate chunks
  - Batch processing with progress tracking
  - Save chunks to disk (JSON/Parquet) for reuse
  - Generate unique chunk IDs (hash-based)

### Phase 3: Retrieval Implementation

- [ ] **3.1** Create `scientific_rag/application/embeddings/encoder.py`

  - Singleton pattern for embedding model
  - Use `intfloat/e5-small-v2`
  - Batch embedding support
  - GPU/CPU device configuration

- [ ] **3.2** Implement `scientific_rag/infrastructure/qdrant.py`

  - Qdrant client wrapper (local Docker or Qdrant Cloud)
  - Collection creation with proper schema
  - Payload fields for metadata filtering:
    - `source`: arxiv or pubmed
    - `section`: introduction, methods, results, conclusion, other
    - `paper_id`: reference to source paper
  - `upsert_chunks(chunks)` - batch insert with embeddings
  - `search(query_vector, filters, k)` - filtered vector search
  - Support for Qdrant filter syntax


- [ ] **3.3** Implement `scientific_rag/application/retrieval/bm25_retriever.py`

  - Use `rank_bm25` library
  - Tokenization with proper preprocessing
  - `search(query, k) -> List[Chunk]` interface
  - Score normalization

- [ ] **3.4** Implement `scientific_rag/application/retrieval/dense_retriever.py`

  - Semantic search using Qdrant
  - Integrate with `QdrantClient` from infrastructure
  - Apply metadata filters from self-query
  - `search(query, filters, k) -> List[Chunk]` interface

- [ ] **3.5** Implement `scientific_rag/application/retrieval/hybrid_retriever.py`
  - Combine BM25 and dense retrieval
  - Pass metadata filters to both retrievers
  - Configurable weights for each method
  - Toggle switches: `use_bm25`, `use_dense`
  - Reciprocal Rank Fusion (RRF) or weighted combination
  - Deduplication of results

### Phase 4: Query Processing & Metadata Filtering

- [ ] **4.1** Implement `scientific_rag/application/query_processing/self_query.py`

  - Extract metadata filters from natural language queries using **rule-based matching**
  - Detect source preferences: "arxiv papers about..." → filter to arxiv
  - Detect section preferences: "in the methods section..." → filter to methods
  - Use regex/keyword matching
  - No LLM needed - metadata is already structured in chunks from dataset
  - Return structured `QueryFilters` object
  - Filters are passed to Qdrant for efficient pre-filtering before vector search

- [ ] **4.2** Implement `scientific_rag/application/query_processing/query_expansion.py`

  - Generate multiple query variations to improve recall
  - Use LLM to create semantically similar queries
  - Configurable `expand_to_n` parameter (default: 3)
  - Example prompt:

    ```
    Generate {n} different versions of this question to search a scientific papers database.
    Each version should capture the same intent but use different wording.
    Separate versions with "###"

    Original: {query}
    ```

  - Search with all expanded queries, merge results
  - Deduplicate before reranking

- [ ] **4.3** Update `scientific_rag/domain/queries.py`

  - Add `QueryFilters` model for self-query results
  - Add `ExpandedQuery` model to hold query variations
  - Example:

    ```python
    class QueryFilters(BaseModel):
        source: Literal["arxiv", "pubmed", "any"] = "any"
        section: Literal["introduction", "methods", "results", "conclusion", "any"] = "any"

    class ExpandedQuery(BaseModel):
        original: str
        variations: list[str]
        filters: QueryFilters | None = None
    ```

### Phase 5: Reranking

- [ ] **5.1** Implement `scientific_rag/application/reranking/cross_encoder.py`
  - Use `cross-encoder/ms-marco-MiniLM-L6-v2` (or similar)
  - `rerank(query, chunks, top_k) -> List[Chunk]` interface
  - Batch processing for efficiency
  - Score-based sorting

### Phase 6: LLM Integration

- [ ] **6.1** Implement `scientific_rag/application/rag/llm_client.py`

  - LiteLLM wrapper for provider abstraction
  - Support for Groq, OpenRouter, OpenAI
  - Configurable model selection
  - Error handling and retries
  - Response streaming (optional)

- [ ] **6.2** Create `scientific_rag/application/rag/prompt_templates.py`

  - RAG prompt template with context injection
  - Citation-aware prompting (instruct model to cite sources)
  - System prompt for scientific Q&A
  - Example:

    ```
    You are a scientific research assistant. Answer the question based on the provided context.
    Always cite your sources using [1], [2], etc.

    Context:
    [1] {chunk_1}
    [2] {chunk_2}
    ...

    Question: {query}

    Answer with citations:
    ```

- [ ] **6.3** Implement `scientific_rag/application/rag/pipeline.py`
  - Main `RAGPipeline` class
  - Orchestrate: Query → Self-Query → Query Expansion → Retrieve (with filters) → Rerank → Generate
  - Full pipeline flow:
    ```
    1. Self-Query: Extract filters (source, section) for Qdrant
    2. Query Expansion: Generate N query variations
    3. Retrieve: Search with all queries (BM25 + Qdrant with filters)
    4. Merge & Deduplicate: Combine results from all queries
    5. Rerank: Cross-encoder scoring
    6. Generate: LLM with citations
    ```
  - Configurable retrieval parameters
  - Toggle for each component: `use_self_query`, `use_query_expansion`, `use_bm25`, `use_dense`, `use_reranking`
  - Citation tracking and formatting

### Phase 7: User Interface

- [ ] **7.1** Create `demo/main.py` with Gradio

  - Text input for questions
  - API key input field (not stored in code)
  - Dropdown for LLM provider/model selection
  - Dropdown for metadata filters (optional manual override):
    - Source: Any / ArXiv / PubMed
    - Section: Any / Introduction / Methods / Results / Conclusion
  - Checkboxes for pipeline components:
    - [ ] Enable Self-Query (metadata extraction)
    - [ ] Enable Query Expansion
    - [ ] Enable BM25
    - [ ] Enable Dense Retrieval (Qdrant)
    - [ ] Enable Reranking
  - Slider for top-k parameter
  - Slider for query expansion count (1-5)
  - Output: Answer with citations
  - Expandable section showing retrieved chunks with metadata

- [ ] **7.2** Add service description

  - Brief explanation of the RAG system
  - Dataset information
  - Usage instructions

- [ ] **7.3** Style and UX improvements
  - Clear layout
  - Loading indicators
  - Error messages for invalid inputs

### Phase 8: Deployment

- [ ] **8.1** Create `requirements.txt` for HuggingFace Spaces

  - Pin versions for reproducibility
  - Note: HF Spaces may need Qdrant Cloud instead of local

- [ ] **8.2** Create HuggingFace Space configuration

  - `README.md` with YAML frontmatter for Gradio SDK
  - Resource requirements (CPU/memory)
  - Configure Qdrant Cloud connection for deployment

- [ ] **8.3** Deploy to HuggingFace Spaces
  - Test with sample queries
  - Verify API key handling
  - Verify Qdrant Cloud connectivity

### Phase 9: Evaluation & Documentation

- [ ] **9.1** Find queries where BM25 outperforms dense retrieval

  - Queries with specific terminology, rare words, or exact phrases
  - Examples:
    - "papers mentioning @xmath0 decay channel"
    - "CLEO detector measurements"

- [ ] **9.2** Find queries where dense retrieval outperforms BM25

  - Semantic similarity queries
  - Paraphrased questions
  - Examples:
    - "How do researchers measure particle lifetimes?"
    - "What methods are used for blood clot prevention?"

- [ ] **9.3** Demonstrate metadata filtering effectiveness

  - Show queries where filtering by source improves results
  - Show queries where filtering by section improves results
  - Examples:
    - "arxiv papers about quantum computing" → filter to arxiv
    - "methodology for clinical trials" → filter to methods section

- [ ] **9.4** Document the system in README.md

  - Architecture overview
  - Installation instructions (including Docker/Qdrant setup)
  - Usage examples
  - Component descriptions
  - Retrieval comparison findings
  - Metadata filtering examples

- [ ] **9.5** Prepare submission materials
  - Source code link
  - Deployed service link
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
description = "Scientific Papers RAG System"
requires-python = ">=3.11"

dependencies = [
    # Data
    "datasets>=3.0.0",
    "huggingface-hub>=0.20.0",

    # ML/Embeddings
    "sentence-transformers>=3.0.0",
    "torch>=2.0.0",
    "numpy>=1.26.0",
    "scipy>=1.11.0",

    # Retrieval
    "rank-bm25>=0.2.2",
    "qdrant-client>=1.8.0",

    # LLM
    "litellm>=1.0.0",

    # Configuration
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",

    # UI
    "gradio>=4.0.0",

    # Utilities
    "loguru>=0.7.0",
    "tqdm>=4.65.0",
    "python-dotenv>=1.0.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
    "pre-commit>=3.0.0",
    "ipykernel>=6.0.0",
]
```

---

## Quick Start Commands

```bash
# Setup
make install

# Run locally
make run-app

# Run tests
make test

# Lint
make lint

# Format
make format
```

---

## Key Implementation Notes

### Chunking Strategy

For scientific papers, consider:

1. **Section-based chunking**: Split by sections first, then by size
2. **Preserve context**: Include section title in each chunk
3. **Handle LaTeX**: Papers contain `@xmath` tokens for math expressions

### Retrieval Comparison

Document specific queries that demonstrate:

- BM25 strength: Exact term matching, rare terminology
- Dense strength: Semantic understanding, paraphrased queries

### LLM Configuration

Recommended free options:

- **Groq**: Fast, free tier with `llama-3.1-8b-instant`
- **OpenRouter**: Multiple model options, some free

### Citation Format

```
Answer: The decay channel measurement shows... [1]. Further analysis using the CLEO detector... [2].

Sources:
[1] "we have studied the leptonic decay..." (arxiv, section: introduction)
[2] "data collected with the CLEO detector..." (arxiv, section: methods)
```

---

## Timeline Suggestion

| Week               | Focus Area                               |
| ------------------ | ---------------------------------------- |
| Week 1 (Dec 9-11)  | Phase 1-2: Setup, Data Loading, Chunking |
| Week 2 (Dec 12-14) | Phase 3-5: Retrieval, Reranking, LLM     |
| Week 3 (Dec 15-16) | Phase 6-8: UI, Deployment, Documentation |

---

## References

- [Assignment Document](./docs/assignment.md)
- [LLM-Engineers-Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook) - Reference architecture
- [Scientific Papers Dataset](https://huggingface.co/datasets/armanc/scientific_papers)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Gradio Documentation](https://www.gradio.app/docs)
