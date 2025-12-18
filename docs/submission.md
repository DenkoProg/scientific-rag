# Здача роботи: Scientific RAG System

## Задача

Створення RAG системи для відповідей на питання про наукові статті з датасетів ArXiv та PubMed з використанням гібридного пошуку та реранкінгу.

## Компоненти системи

### Джерело даних
**armanc/scientific_papers** (Hugging Face Datasets) - ArXiv та PubMed наукові статті з розділами: Introduction, Methods, Results, Conclusion.

### Chunking
- Розмір: 512 токенів
- Overlap: 25 токенів
- Мінімальний розмір: 25 токенів
- Метадані: source (arxiv/pubmed), section, paper_id

### LLM
- **Провайдер**: Groq (за замовчуванням), OpenRouter
- **Моделі**: Llama 3.1 8B, Llama 3.3 70B, Qwen, Nova
- **Інтеграція**: LiteLLM для уніфікованого API
- API ключ вводиться користувачем через UI

### Retriever
**Гібридний пошук** (можна вмикати/вимикати кожен):
- **BM25**: Keyword-based пошук через Qdrant sparse vectors
- **Dense**: Semantic пошук з embedding моделлю `intfloat/e5-small-v2`
- **Query Expansion**: Генерація варіацій запиту через LLM (1-5 варіантів)
- **Self-Query**: Автоматичне визначення фільтрів з запиту

**Векторна БД**: Qdrant Cloud

### Reranker
**Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Ре-ранкінг top-K результатів (1-20 чанків)
- Можна вимкнути через UI

### Citations
- Номери джерел в квадратних дужках [1], [2]
- Відображення retrieved chunks з метаданими (source, section, paper_id, score)
- Порівняння BM25 vs Dense результатів

### UI
**Gradio** з можливостями:
- Вибір провайдера та моделі
- Конфігурація параметрів (top-k, expansion count)
-Toggling компонентів pipeline
- Відображення метрик (час виконання, кількість чанків)
- Example questions

### Інше
- **DDD архітектура**: domain, application, infrastructure layers
- **Type safety**: Pydantic моделі
- **Logging**: Loguru
- **Deployment**: HF Spaces ready (auto-detect, in-memory Qdrant)
- **CLI**: Make commands для chunk-data, index-qdrant, run-app

## Виконання

**Індивідуальне виконання**: Denys Koval

## Посилання

- **Deployed Service**: https://huggingface.co/spaces/ai-department-lpnu/scientific-rag
- **Source Code**: https://github.com/DenkoProg/scientific-rag

## Інструкції з запуску

### Локально
```bash
# 1. Clone та встановити залежності
git clone https://github.com/DenkoProg/scientific-rag.git
cd scientific-rag
make install

# 2. Налаштувати .env
cp .env.dist .env
# Додати API ключі: LLM_API_KEY, QDRANT_URL, QDRANT_API_KEY

# 3. Обробити дані та створити індекс
make pipeline

# 4. Запустити додаток
make run-app
```

### HF Spaces
Автоматичний deploy з main branch - всі необхідні файли (app.py, requirements.txt) в root директорії.
