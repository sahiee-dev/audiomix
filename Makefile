.PHONY: help dev-backend dev-frontend install lint test clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Installation ─────────────────────────────────────────────────────────────

install: install-backend install-frontend ## Install all dependencies

install-backend: ## Install Python backend dependencies
	cd backend && pip install -r app/requirements.txt

install-frontend: ## Install Node.js frontend dependencies
	cd frontend && npm install

# ─── Development servers ──────────────────────────────────────────────────────

dev-backend: ## Start the FastAPI backend on port 8000 (hot-reload)
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend: ## Start the Vite dev server on port 5173
	cd frontend && npm run dev

# ─── Quality ──────────────────────────────────────────────────────────────────

lint: ## Run ruff linter on backend code
	cd backend && ruff check app/

test: ## Run backend unit tests with pytest
	cd backend && pytest app/tests/ -v

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove uploaded audio files and Python caches
	rm -rf /tmp/smart_mix_uploads
	find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find backend -name "*.pyc" -delete 2>/dev/null || true
