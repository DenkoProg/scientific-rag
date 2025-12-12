# Team Roles & Task Distribution

> **Project**: Scientific Advanced RAG System
> **Team Size**: 3 members
> **Timeline**: December 12-16, 2025
> **Strategy**: Parallel development with clear ownership and minimal dependencies

---

## 👥 Team Structure

### Member 1: Data Pipeline Lead
**Focus**: Data processing, embeddings, vector database infrastructure

### Member 2: Retrieval Engineer
**Focus**: Search algorithms, BM25, dense retrieval, reranking

### Member 3: LLM & Integration Lead
**Focus**: Query processing, LLM integration, RAG pipeline, UI

---

## 📋 Detailed Task Assignments

### 🔹 Member 1: Data Pipeline Lead

#### Phase 2: Chunking Strategy (Priority: HIGH)
- **2.1** Create `scientific_rag/application/chunking/base.py`
  - Abstract `BaseChunker` class
  - Define interface: `chunk(document) -> List[Chunk]`

- **2.2** Implement `scientific_rag/application/chunking/scientific_chunker.py`
  - Section-aware chunking with metadata preservation
  - Normalize section names to enum values
  - Handle LaTeX tokens (@xmath)

- **2.3** Create processing script to generate chunks
  - Batch processing with progress tracking
  - Save to `data/processed/` as JSON/Parquet
  - Generate hash-based chunk IDs

#### Phase 3: Embeddings & Vector Database (Priority: HIGH)
- **3.1** Create `scientific_rag/application/embeddings/encoder.py`
  - Singleton pattern for `intfloat/e5-small-v2`
  - Batch embedding support (batch_size=32)
  - CPU/GPU device configuration

- **3.3** Implement `scientific_rag/infrastructure/qdrant.py`
  - Qdrant client wrapper (local Docker + cloud support)
  - Collection creation with schema (384-d vectors)
  - Metadata payload: source, section, paper_id, position
  - `upsert_chunks(chunks)` with embeddings
  - `search(query_vector, filters, k)` with filtering

#### Deliverables
- Working chunking pipeline that processes papers → chunks with metadata
- Qdrant collection populated with embedded chunks
- Script to run: `python scripts/process_and_index.py`

**Estimated Time**: 2-3 days

---

### 🔹 Member 2: Retrieval Engineer

#### Phase 3: Retrieval Implementation (Priority: HIGH)
- **3.2** Implement `scientific_rag/application/retrieval/bm25_retriever.py`
  - Use `rank_bm25` library
  - Tokenization with preprocessing
  - `search(query, k) -> List[Chunk]` interface
  - Score normalization

- **3.4** Implement `scientific_rag/application/retrieval/dense_retriever.py`
  - Semantic search using Qdrant (depends on Member 1's 3.3)
  - Apply metadata filters from `QueryFilters`
  - `search(query, filters, k) -> List[Chunk]`

- **3.5** Implement `scientific_rag/application/retrieval/hybrid_retriever.py`
  - Combine BM25 + dense retrieval
  - Reciprocal Rank Fusion (RRF) or weighted combination
  - Configurable weights: `bm25_weight`, `dense_weight`
  - Toggle switches: `use_bm25`, `use_dense`
  - Deduplication logic

#### Phase 5: Reranking (Priority: MEDIUM)
- **5.1** Implement `scientific_rag/application/reranking/cross_encoder.py`
  - Use `cross-encoder/ms-marco-MiniLM-L6-v2`
  - `rerank(query, chunks, top_k) -> List[Chunk]`
  - Batch processing for efficiency
  - Score-based sorting

#### Phase 9: Evaluation Support (Priority: LOW)
- **9.1** Find BM25-best queries
  - Document specific terminology queries
  - Exact phrase matching examples

- **9.2** Find dense-best queries
  - Semantic similarity queries
  - Paraphrased questions

#### Deliverables
- BM25 retriever (can test standalone with chunks)
- Dense retriever (integrates with Member 1's Qdrant)
- Hybrid retriever combining both
- Reranker module
- Comparison analysis for BM25 vs Dense

**Estimated Time**: 2-3 days

---

### 🔹 Member 3: LLM & Integration Lead

#### Phase 4: Query Processing (Priority: HIGH)
- **4.1** Implement `scientific_rag/application/query_processing/self_query.py`
  - Rule-based metadata filter extraction
  - Regex/keyword matching for source (arxiv/pubmed)
  - Pattern matching for section (introduction/methods/results/conclusion)
  - Return `QueryFilters` object

- **4.2** Implement `scientific_rag/application/query_processing/query_expansion.py`
  - LLM-based query variation generation
  - Configurable `expand_to_n` parameter (default: 3)
  - Deduplicate expanded queries

- **4.3** Update `scientific_rag/domain/queries.py`
  - Already done, verify completeness

#### Phase 6: LLM Integration (Priority: HIGH)
- **6.1** Implement `scientific_rag/application/rag/llm_client.py`
  - LiteLLM wrapper for OpenRouter
  - Support `openai/gpt-oss-120b:free`
  - Error handling and retries
  - Optional: response streaming

- **6.2** Create `scientific_rag/application/rag/prompt_templates.py`
  - RAG prompt with context injection
  - Citation-aware prompting ([1], [2] format)
  - System prompt for scientific Q&A

- **6.3** Implement `scientific_rag/application/rag/pipeline.py`
  - Main `RAGPipeline` orchestration class
  - Full flow: Self-Query → Query Expansion → Retrieve → Rerank → Generate
  - Toggle switches for each component
  - Citation tracking

#### Phase 7: User Interface (Priority: MEDIUM)
- **7.1** Create `demo/main.py` with Gradio
  - Text input for questions
  - API key input field
  - Dropdown for model selection
  - Metadata filter dropdowns (source, section)
  - Component toggle checkboxes
  - Top-k slider, expansion count slider
  - Output: Answer with citations + retrieved chunks

- **7.2** Add service description
  - RAG system explanation
  - Dataset info (320K papers)

- **7.3** Style and UX improvements
  - Clean layout with loading indicators
  - Error messages

#### Phase 8: Deployment (Priority: LOW)
- **8.1** Create `requirements.txt` for HuggingFace Spaces
- **8.2** HuggingFace Space configuration (`README.md` with YAML)
- **8.3** Deploy and test

#### Phase 9: Documentation (Priority: LOW)
- **9.3** Demonstrate metadata filtering effectiveness
- **9.4** Document system in `README.md`
- **9.5** Prepare submission materials

#### Deliverables
- Query processing modules (self-query, expansion)
- LLM client with prompt templates
- Complete RAG pipeline
- Gradio UI demo
- Documentation and deployment

**Estimated Time**: 3-4 days

---

## 🔄 Integration Points & Dependencies

### Critical Path
```
Day 1-2:
  Member 1: Chunking (2.1, 2.2, 2.3) → Embeddings (3.1)
  Member 2: BM25 (3.2) [can start immediately]
  Member 3: Self-query (4.1), LLM client (6.1), Prompts (6.2)

Day 2-3:
  Member 1: Qdrant client (3.3) + Index chunks [BLOCKER for Member 2's 3.4]
  Member 2: Dense retriever (3.4) [WAIT for 3.3] → Hybrid (3.5)
  Member 3: Query expansion (4.2), Pipeline stub (6.3)

Day 3-4:
  Member 1: Support/testing, optimize indexing
  Member 2: Reranking (5.1) → Integration testing
  Member 3: Complete Pipeline (6.3) → Gradio UI (7.1)

Day 4-5:
  All: Integration testing, bug fixes
  Member 3: UI polish (7.2, 7.3), Deployment (8.1, 8.2, 8.3)
  Member 1 & 2: Evaluation (9.1, 9.2, 9.3)
  Member 3: Documentation (9.4, 9.5)
```

### Key Handoffs
1. **Member 1 → Member 2**: Qdrant client ready (Day 2)
2. **Member 1 & 2 → Member 3**: Retrievers ready for pipeline (Day 3)
3. **Member 3 → All**: Pipeline ready for testing (Day 3-4)

---

## ✅ Success Criteria

### By December 14 (Mid-checkpoint)
- [ ] Chunks generated and saved to disk (Member 1)
- [ ] Qdrant collection created and indexed (Member 1)
- [ ] BM25 retriever working (Member 2)
- [ ] Dense retriever working (Member 2)
- [ ] LLM client + prompts ready (Member 3)

### By December 16 (Final Deadline)
- [ ] Complete RAG pipeline functional
- [ ] Gradio UI deployed locally
- [ ] Evaluation examples documented
- [ ] README.md with usage instructions
- [ ] Ready for HuggingFace Spaces deployment

---

## 🚨 Risk Mitigation

### Risk: Qdrant indexing takes longer than expected
**Mitigation**: Member 1 starts with small sample (1K papers), scales up gradually

### Risk: Dense retriever blocked by Qdrant
**Mitigation**: Member 2 prioritizes BM25 + Reranking first (no dependencies)

### Risk: LLM API rate limits
**Mitigation**: Member 3 implements retry logic + fallback prompts, tests with small queries

### Risk: Integration issues at Day 3
**Mitigation**: Daily integration checkpoints, mock interfaces early

---

## 📚 Quick Reference

### Useful Make Commands
```bash
make install        # Install dependencies
make qdrant-up      # Start Qdrant
make qdrant-down    # Stop Qdrant
make format         # Format code
make lint           # Check code quality
```

---

**Good luck team! 🚀**
