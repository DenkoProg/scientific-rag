---
title: Scientific RAG System
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: demo/main.py
pinned: false
license: mit
tags:
  - rag
  - scientific-papers
  - arxiv
  - pubmed
  - retrieval
  - question-answering
python_version: 3.11
---

# 🔬 Scientific RAG System

**Retrieval-Augmented Generation for Scientific Papers**

This system allows you to ask questions about scientific papers from ArXiv and PubMed datasets, receiving answers with citations from source documents.

## 🌟 Features

- 🔍 **Hybrid Retrieval**: Combines BM25 (keyword-based) and Dense (semantic) search
- 🎯 **Query Processing**: Self-query metadata extraction and query expansion
- 📊 **Reranking**: Cross-encoder model for improved relevance
- 🏷️ **Metadata Filtering**: Filter by source and section type
- 📝 **Citations**: Answers include source citations
- 🎨 **Interactive UI**: Configurable pipeline components

## 🚀 Quick Start

1. Enter your API key (OpenRouter or Groq) in the settings
2. Type your question about scientific papers
3. Adjust retrieval settings if needed
4. Get answers with citations!

## 🔑 API Keys

Get free API keys from:
- [OpenRouter](https://openrouter.ai/) - Multiple free models available
- [Groq](https://console.groq.com/) - Fast inference with free tier

## 🏗️ Architecture

```
Query → Processing → Hybrid Retrieval → Reranking → LLM Generation → Answer + Citations
```

## 📚 Dataset

Uses the [armanc/scientific_papers](https://huggingface.co/datasets/armanc/scientific_papers) dataset (ArXiv subset).

## 💡 Example Questions

- "What are the main approaches to quantum error correction?"
- "Explain the mechanisms of plasma confinement"
- "How do systems enable precise gene editing?"
- "What role do protein folding dynamics play in neurodegenerative diseases?"

## 📄 License

MIT License

## 🔗 Links

- [GitHub Repository](https://github.com/DenkoProg/scientific-rag)
- [Dataset](https://huggingface.co/datasets/armanc/scientific_papers)
