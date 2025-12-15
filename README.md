# Scientific RAG

🔬 **Retrieval-Augmented Generation System for Scientific Papers**

A production-ready RAG (Retrieval-Augmented Generation) system for answering questions about scientific papers using the [armanc/scientific_papers](https://huggingface.co/datasets/armanc/scientific_papers) dataset (ArXiv + PubMed).

## 🌟 Features

- 🔍 **Hybrid Retrieval**: Combines BM25 (keyword-based) and Dense (semantic) search
- 🎯 **Query Processing**: Self-query metadata extraction and query expansion
- 📊 **Reranking**: Cross-encoder model for improved relevance
- 🏷️ **Metadata Filtering**: Filter by source (ArXiv/PubMed) and section (Intro/Methods/Results/Conclusion)
- 📝 **Citations**: Answers include source citations
- 🎨 **Interactive UI**: Gradio-based web interface with configurable pipeline components
- 🗄️ **Vector Database**: Qdrant for efficient similarity search

## 🏗️ Architecture

```
User Query
    ↓
┌─────────────────────────────────────┐
│ 1. Query Processing                 │
│    - Self-Query (metadata filters)  │
│    - Query Expansion (variations)   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Hybrid Retrieval                 │
│    - BM25 (keyword search)          │
│    - Dense (semantic search)        │
│    - Metadata filtering             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Reranking                        │
│    - Cross-encoder scoring          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. Answer Generation                │
│    - LLM with context injection     │
│    - Citation-aware prompting       │
└─────────────────────────────────────┘
    ↓
Answer with Citations
```

## ⚙️ Installation

### 🔧 Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- UV package manager

### 📦 Setup Steps

#### 1. Clone the repository

```bash
git clone https://github.com/DenkoProg/scientific-rag.git
cd scientific-rag
```

#### 2. Install `uv` — A fast Python package manager

📖 [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 3. Install dependencies

```bash
make install
```

This will set up the virtual environment, install dependencies, and configure pre-commit hooks automatically.

#### 4. Configure environment

Copy `.env` template and add your API keys:

```bash
cp .env.dist .env
# Edit .env and add your LLM API key

**Get API Keys:**
- [OpenRouter](https://openrouter.ai/) - Multiple free models
- [Groq](https://console.groq.com/) - Free tier
- [OpenAI](https://platform.openai.com/) - Paid

## 🚀 Quick Start

### 1. Start Qdrant Vector Database

```bash
make qdrant-up
```

Qdrant will be available at `http://localhost:6333`

### 2. Process Data

```bash
# Complete pipeline: chunk papers + index to Qdrant
make pipeline

# Or run steps separately:
make chunk-data      # Generate chunks from papers (10-30 min)
make index-qdrant    # Embed and index to Qdrant (20-40 min)
```

**Note**: Processing time depends on `DATASET_SAMPLE_SIZE` in `.env`:
- `1000` papers ≈ 10 minutes
- `10000` papers ≈ 30 minutes
- Full dataset (200K+) ≈ several hours

### 3. Launch Web Interface

```bash
make run-app
```

Open `http://localhost:7860` in your browser.

### 4. Ask Questions!

Try example queries:
- "What are the main approaches to quantum error correction?"
- "How do systems enable precise gene editing?"
- "Explain plasma confinement mechanisms in tokamaks"
- "What role do protein folding dynamics play in neurodegenerative diseases?"

## 📖 Usage Guide

### Web Interface

**Configuration Options:**

1. **API Settings**
   - Enter your LLM API key (not stored)
   - Select provider (Groq/OpenRouter/OpenAI)
   - Choose model

2. **Metadata Filters**
   - Source: Any / ArXiv / PubMed
   - Section: Any / Introduction / Methods / Results / Conclusion

3. **Pipeline Components** (toggle on/off)
   - ☑ Self-Query: Auto-extract filters from query
   - ☑ Query Expansion: Generate query variations
   - ☑ BM25: Keyword-based retrieval
   - ☑ Dense: Semantic vector search
   - ☑ Reranking: Cross-encoder scoring

4. **Parameters**
   - Top-K: Number of chunks to retrieve (1-20)
   - Query Expansion Count: Number of variations (1-5)

### Command Line Interface

```bash
# View system info
make info

# Process specific dataset split
DATASET_SPLIT=pubmed make chunk-data

# Custom chunk size
CHUNK_SIZE=256 make chunk-data
```

## 🔧 Configuration

All settings in `.env`:

```bash
# Dataset
DATASET_NAME=armanc/scientific_papers
DATASET_SPLIT=arxiv              # arxiv or pubmed
DATASET_SAMPLE_SIZE=10000        # None for full dataset

# Chunking
CHUNK_SIZE=512                   # words per chunk
CHUNK_OVERLAP=50                 # overlap between chunks

# Retrieval
RETRIEVAL_TOP_K=10
BM25_WEIGHT=0.5
DENSE_WEIGHT=0.5

# Reranking
RERANK_TOP_K=5

# Embeddings
EMBEDDING_MODEL_NAME=intfloat/e5-small-v2
EMBEDDING_DEVICE=cpu             # or cuda/mps

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=scientific_papers
```

## 🚀 Deployment

Want to deploy for free? See the [ops/](ops/) folder:
- **[ops/QUICK_START.md](ops/QUICK_START.md)** - Deploy in 3 commands
- **[ops/DEPLOYMENT.md](ops/DEPLOYMENT.md)** - Full deployment guide
- **[ops/README_SPACES.md](ops/README_SPACES.md)** - HF Spaces README template

## 🎯 Retrieval Comparison

### When BM25 Outperforms Dense

**Best for: Exact terms, rare words, acronyms**

Examples:
- "papers mentioning @xmath0 decay channel"
- "CLEO detector measurements"
- "CRISPR-Cas9 mechanism"

### When Dense Outperforms BM25

**Best for: Semantic similarity, paraphrased questions**

Examples:
- "How do cells divide?" (finds "mitosis", "cellular division")
- "Methods for preventing blood clots" (finds "anticoagulation therapy")