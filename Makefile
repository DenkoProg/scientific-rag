# Scientific RAG - Makefile

.PHONY: install
install: ## Install dependencies and setup pre-commit hooks
	@echo "🚀 Installing dependencies from lockfile"
	@uv sync --frozen
	@uv run pre-commit install

.PHONY: qdrant-up
qdrant-up: ## Start Qdrant vector database
	@echo "🚀 Starting Qdrant..."
	@docker compose up -d qdrant
	@echo "✅ Qdrant running at http://localhost:6333"

.PHONY: qdrant-down
qdrant-down: ## Stop Qdrant vector database
	@echo "🛑 Stopping Qdrant..."
	@docker compose down

.PHONY: qdrant-logs
qdrant-logs: ## Show Qdrant logs
	@docker compose logs -f qdrant

.PHONY: chunk-data
chunk-data: ## Process and chunk papers
	@uv run cli chunk

.PHONY: index-qdrant
index-qdrant: ## Index chunks to Qdrant
	@uv run cli index

.PHONY: pipeline
pipeline: ## Run complete pipeline (chunk + index)
	@uv run cli pipeline

.PHONY: info
info: ## Show pipeline configuration and Qdrant status
	@uv run cli info

.PHONY: run-app
run-app: ## Run Gradio application
	@uv run python app.py

.PHONY: lint
lint: ## Run ruff linter
	@uv run ruff check

.PHONY: format
format: ## Format code and fix linting issues
	@uv run ruff format
	@uv run ruff check --fix

.PHONY: clean
clean: ## Clean cache and temporary files
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +

.PHONY: help
help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help