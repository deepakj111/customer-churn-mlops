# Makefile — put this at the project root

.PHONY: help setup install format lint test test-unit test-integration train serve dashboard drift-report save-reference docker-build docker-up docker-down clean

help:   ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: install   ## Full first-time setup
	poetry run pre-commit install
	poetry run dvc pull

install:   ## Install all dependencies
	poetry install

format:   ## Auto-format code with black + isort
	poetry run black src/ tests/ scripts/ dashboards/
	poetry run isort src/ tests/ scripts/ dashboards/

lint:   ## Lint code with flake8 + mypy
	poetry run flake8 src/ tests/ scripts/
	poetry run mypy src/

test:   ## Run all tests with coverage
	poetry run pytest

test-unit:   ## Run only unit tests
	poetry run pytest tests/unit/

test-integration:   ## Run only integration tests
	poetry run pytest tests/integration/

train:   ## Run full training pipeline
	poetry run python main.py

serve:   ## Start the FastAPI server locally
	poetry run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

dashboard:   ## Launch Streamlit dashboard
	poetry run streamlit run dashboards/streamlit_app.py

drift-report:   ## Generate latest drift monitoring report
	poetry run python scripts/generate_drift_report.py

save-reference:   ## Save training data as drift reference snapshot
	poetry run python scripts/save_reference_data.py

feast-apply:   ## Apply Feast Feature Store registry
	cd src/features/feast_repo && poetry run feast apply

feast-materialize:   ## Materialize Feast online store
	cd src/features/feast_repo && poetry run feast materialize-incremental $$(date -Iseconds)

docker-build:   ## Build all Docker images
	docker compose -f docker/docker-compose.yml build

docker-up:   ## Start full local stack with Docker Compose
	docker compose -f docker/docker-compose.yml up -d

docker-down:   ## Stop all Docker containers
	docker compose -f docker/docker-compose.yml down

clean:   ## Remove Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
