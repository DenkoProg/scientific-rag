# Team Roles & Task Distribution

> **Project**: Scientific Advanced RAG System
> **Team Size**: 3 members
> **Timeline**: December 12-16, 2025
> **Strategy**: Parallel development with clear ownership and minimal dependencies

---

## 👥 Team Structure

### Member 1: Data Pipeline Lead
**Focus**: Data processing, chunking, embeddings, vector database infrastructure

### Member 2: Retrieval Engineer
**Focus**: BM25, dense retrieval, hybrid search, reranking

### Member 3: LLM & Integration Lead
**Focus**: Query processing, LLM integration, RAG pipeline, UI

---

## 📋 Detailed Task Assignments

### 🔹 Member 1: Data Pipeline Lead

#### Phase 1: Project Setup (Priority: HIGH)
- **1.1** Update `pyproject.toml` with dependencies
  - datasets, sentence-transformers, fastembed, qdrant-client
  - litellm, gradio, pydantic-settings, typer, tenacity

- **1.2** Create `docker-compose.yaml` for Qdrant

- **1.3** Create `src/scientific_rag/settings.py`
  - Pydantic BaseSettings with all configuration
  - Qdrant, embedding, chunking, retrieval, LLM settings

- **1.4** Create `src/scientific_rag/domain/` entities
  - `types.py`: DataSource, SectionType enums
  - `documents.py`: ScientificPaper, PaperChunk models
  - `responses.py`: RAGResponse model

- **1.5** Implement `src/scientific_rag/application/data_loader.py`
  - HuggingFace dataset loading with tqdm progress

- **1.6** Create `src/scientific_rag/cli.py` with Typer
  - `chunk`, `index`, `pipeline`, `info` commands

#### Phase 2: Chunking Strategy (Priority: HIGH)
- **2.1** Implement `src/scientific_rag/application/chunking/scientific_chunker.py`
  - Paragraph-based chunking with word-based sizing
  - Abstract handling, section inference by position
  - LaTeX normalization, hash-based chunk IDs

- **2.2** Create `src/scientific_rag/scripts/chunk_data.py`
  - Batch processing, stream-write to JSON
  - Output to `data/processed/chunks_{split}.json`

#### Phase 3: Embeddings & Vector Database (Priority: HIGH)
- **3.1** Create `src/scientific_rag/application/embeddings/encoder.py`
  - Singleton pattern for `intfloat/e5-small-v2`
  - Query/passage prefixing for E5 models
  - Batch embedding support (batch_size=32)

- **3.2** Implement `src/scientific_rag/infrastructure/qdrant.py`
  - QdrantService class with sync client
  - Hybrid collection: dense (384-d) + sparse BM25 vectors
  - Named vectors: "dense" and "bm25"
  - `upsert_chunks()`, `search_dense()`, `search_sparse()`
  - Payload indexes on source, section, paper_id

- **3.3** Create `src/scientific_rag/scripts/index_qdrant.py`
  - Load chunks from JSON, embed, upload to Qdrant
  - Batch processing with progress tracking

#### Deliverables
- Working chunking pipeline: papers → chunks with metadata
- Qdrant collection with dense + sparse vectors
- CLI commands: `make chunk-data`, `make index-qdrant`, `make pipeline`

**Estimated Time**: 2-3 days

---

### 🔹 Member 2: Retrieval Engineer

#### Phase 3: Retrieval Implementation (Priority: HIGH)
- **3.4** Implement `src/scientific_rag/application/retrieval/bm25_retriever.py`
  - Use `fastembed.SparseTextEmbedding` with `Qdrant/bm25` model
  - Sparse vector search via Qdrant (native filtering support)
  - Score normalization (min-max scaling)
  - `search(query, k, filters) -> List[PaperChunk]`

- **3.5** Implement `src/scientific_rag/application/retrieval/dense_retriever.py`
  - Semantic search using Qdrant dense vectors
  - Use shared `encoder` singleton for query embedding
  - Apply metadata filters from QueryFilters
  - `search(query, k, filters) -> List[PaperChunk]`

- **3.6** Implement `src/scientific_rag/application/retrieval/hybrid_retriever.py`
  - Combine BM25 + dense retrieval with weighted scoring
  - Configurable weights: `bm25_weight`, `dense_weight`
  - Toggle switches: `use_bm25`, `use_dense`
  - Merge results by chunk_id, sum weighted scores

#### Phase 5: Reranking (Priority: MEDIUM)
- **5.1** Implement `src/scientific_rag/application/reranking/cross_encoder.py`
  - Use `cross-encoder/ms-marco-MiniLM-L6-v2`
  - Singleton pattern with lazy loading
  - `rerank(query, chunks, top_k) -> List[PaperChunk]`
  - Copy chunks before modifying scores

#### Phase 9: Evaluation Support (Priority: LOW)
- **9.1** Find BM25-best queries
  - Specific terminology, rare words, exact phrases
  - Examples: "@xmath0 decay", "CLEO detector"

- **9.2** Find dense-best queries
  - Semantic similarity, paraphrased questions
  - Examples: "How do researchers measure particle lifetimes?"

#### Deliverables
- BM25 retriever with Qdrant sparse search
- Dense retriever with Qdrant dense search
- Hybrid retriever combining both
- Reranker module
- Comparison analysis for BM25 vs Dense

**Estimated Time**: 2-3 days

---

### 🔹 Member 3: LLM & Integration Lead

#### Phase 4: Query Processing (Priority: HIGH)
- **4.1** Implement `src/scientific_rag/application/query/query_processor.py` (Unified module)
  - **Self-Query**: Rule-based regex for source/section detection
  - **Query Expansion**: LLM-based variations (configurable expand_to_n)
  - `process(query, use_expansion, extract_filters) -> ExpandedQuery`

- **4.2** Update `src/scientific_rag/domain/queries.py`
  - `QueryFilters` with `to_qdrant_filter()` method
  - `ExpandedQuery` with `all_queries()` helper
  - `Query`, `EmbeddedQuery` models

#### Phase 6: LLM Integration (Priority: HIGH)
- **6.1** Implement `src/scientific_rag/application/rag/llm_client.py`
  - LiteLLM wrapper for OpenRouter, Groq
  - Retry logic with tenacity (3 attempts, exponential backoff)
  - Dynamic API key/model override from UI
  - Singleton pattern

- **6.2** Create `src/scientific_rag/application/rag/prompt_templates.py`
  - `RAGPrompts` class with static methods
  - `SYSTEM_PROMPT`, `format_context()`, `generate_rag_prompt()`
  - Citation-aware formatting with [1], [2] labels

- **6.3** Implement `src/scientific_rag/application/rag/pipeline.py`
  - `RAGPipeline` class orchestrating all components
  - `run()` with toggle parameters for each component
  - Return `RAGResponse` with execution metadata

#### Phase 7: User Interface (Priority: MEDIUM)
- **7.1** Create `app.py` (root level) with Gradio
  - `RAGPipelineWrapper` for UI-specific logic
  - Provider/model dropdowns, component toggles
  - Sliders for top-k, expansion count, display chunks
  - Tabbed output: Answer + Retrieved Chunks
  - Example queries

- **7.2** Add service description and header

- **7.3** Style and UX improvements
  - Input validation, error messages
  - Loading states, clear button

#### Phase 8: Deployment (Priority: LOW)
- **8.1** Create `requirements.txt` for HuggingFace Spaces
- **8.2** HuggingFace Space configuration (README.md YAML)
- **8.3** Deploy and test with Qdrant Cloud

#### Phase 9: Documentation (Priority: LOW)
- **9.3** Demonstrate metadata filtering effectiveness
- **9.4** Document system in `README.md`
- **9.5** Prepare submission materials

#### Deliverables
- Query processing module (unified self-query + expansion)
- LLM client with prompt templates
- Complete RAG pipeline
- Gradio UI in `app.py`
- Documentation and deployment

**Estimated Time**: 3-4 days

---

## 🔄 Integration Points & Dependencies

### Critical Path
```
Day 1-2:
  Member 1: Setup (1.1-1.6) → Chunking (2.1, 2.2) → Embeddings (3.1)
  Member 2: BM25 retriever (3.4) [can start with mock data]
  Member 3: Query processor (4.1), LLM client (6.1), Prompts (6.2)

Day 2-3:
  Member 1: Qdrant service (3.2) → Index script (3.3) [BLOCKER for Member 2]
  Member 2: Dense retriever (3.5) [WAIT for 3.2] → Hybrid (3.6)
  Member 3: Pipeline stub (6.3), domain models (4.2)

Day 3-4:
  Member 1: Support/testing, optimize indexing
  Member 2: Reranking (5.1) → Integration testing
  Member 3: Complete Pipeline (6.3) → Gradio UI (7.1)

Day 4-5:
  All: Integration testing, bug fixes
  Member 3: UI polish (7.2, 7.3), Deployment (8.1-8.3)
  Member 1 & 2: Evaluation (9.1, 9.2, 9.3)
  Member 3: Documentation (9.4, 9.5)
```

### Key Handoffs
1. **Member 1 → Member 2**: Qdrant service ready (Day 2)
2. **Member 1 & 2 → Member 3**: Retrievers ready for pipeline (Day 3)
3. **Member 3 → All**: Pipeline ready for testing (Day 3-4)

---

## 🚨 Risk Mitigation

### Risk: Qdrant indexing takes longer than expected
**Mitigation**: Member 1 starts with small sample (1K papers), scales up gradually

### Risk: Dense retriever blocked by Qdrant
**Mitigation**: Member 2 prioritizes BM25 + Reranking first (no dependencies)

### Risk: LLM API rate limits
**Mitigation**: Member 3 implements retry logic with tenacity, uses free tier models

### Risk: Integration issues at Day 3
**Mitigation**: Daily integration checkpoints, shared domain models early

---

## 📚 Quick Reference

### Useful Make Commands
```bash
make install        # Install dependencies with UV
make qdrant-up      # Start Qdrant Docker container
make qdrant-down    # Stop Qdrant
make chunk-data     # Process papers into chunks
make index-qdrant   # Index chunks to Qdrant
make pipeline       # Run complete data pipeline
make run-app        # Start Gradio UI
make info           # Show configuration and status
make format         # Format code with ruff
make lint           # Check code quality
make clean          # Clean cache files
```

### CLI Commands
```bash
uv run cli chunk --batch-size 10000
uv run cli index --embedding-batch-size 32 --upload-batch-size 100
uv run cli pipeline
uv run cli info
```

### Project Structure
```
src/scientific_rag/
├── cli.py                    # Typer CLI
├── settings.py               # Pydantic settings
├── domain/                   # Core models
├── application/
│   ├── data_loader.py
│   ├── chunking/
│   ├── embeddings/
│   ├── query/                # Unified query processing
│   ├── retrieval/
│   ├── reranking/
│   └── rag/
├── infrastructure/
│   └── qdrant.py
└── scripts/
    ├── chunk_data.py
    └── index_qdrant.py
```

---

**Good luck team! 🚀**
