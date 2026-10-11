# Contributing to Customer Churn MLOps

Thank you for your interest in contributing to the Customer Churn MLOps platform! We welcome contributions from data scientists, machine learning engineers, and software developers.

This document provides guidelines for environment setup, coding standards, testing practices, and our pull request process.

---

## Table of Contents
1. [Development Environment Setup](#development-environment-setup)
2. [Workflow & Branching Strategy](#workflow--branching-strategy)
3. [Code Standards & Linting](#code-standards--linting)
4. [Testing & Quality Assurance](#testing--quality-assurance)
5. [Documentation Guidelines](#documentation-guidelines)
6. [Submitting a Pull Request](#submitting-a-pull-request)

---

## Development Environment Setup

### Prerequisites
- **Python**: 3.11.x (managed via `pyenv` or native package manager)
- **Poetry**: Dependency and virtual environment manager (`>= 1.8.0`)
- **Docker & Docker Compose**: For containerized serving and MLflow orchestration
- **Git & DVC**: For code and dataset version control

### Initializing the Project
```bash
# 1. Clone repository
git clone https://github.com/deepakj111/customer-churn-mlops.git
cd customer-churn-mlops

# 2. Install dependencies with Poetry (including dev & test groups)
poetry install

# 3. Pull versioned datasets via DVC
poetry run dvc pull

# 4. Copy environment configuration
cp .env.example .env
```

---

## Workflow & Branching Strategy

We follow a GitHub Flow model with descriptive branch names:
- `feature/<feature-name>`: New feature implementations (e.g., `feature/conformal-risk-sets`)
- `fix/<bug-description>`: Bug fixes (e.g., `fix/retraining-force-flag`)
- `refactor/<component>`: Code refactoring without behavior change
- `docs/<doc-topic>`: Documentation updates

### Commit Messages
Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat: add async batch prediction endpoint with job polling`
- `fix: correct false positive cost alignment in model_config.yaml`
- `test: add integration test suite for /readiness probe`
- `docs: update deployment architecture diagram`

---

## Code Standards & Linting

We maintain strict production code standards:
- **Type Annotations**: Mandatory type hints across all function signatures (`from __future__ import annotations`).
- **Formatting & Linting**: We use **Ruff** for linting and code formatting.
- **Error Handling**: Use structured exceptions; never leak raw internal tracebacks through API endpoints.
- **Logging**: Use `src.utils.logging.get_logger(__name__)`. Do not use naked `print()` statements in library code.

### Running Linters Locally
```bash
# Format and lint with ruff
poetry run ruff check . --fix
poetry run ruff format .
```

---

## Testing & Quality Assurance

All contributions must include comprehensive automated tests:
- **Unit Tests**: Located in `tests/unit/` (target individual transformations, metrics, threshold optimization).
- **Integration Tests**: Located in `tests/integration/` (target FastAPI endpoints, pipeline orchestrations).
- **Data Validation Tests**: Located in `tests/data_tests/` (Pandera schema constraints).

### Running Tests
```bash
# Run full test suite with coverage report
poetry run pytest

# Run API integration tests only
poetry run pytest tests/integration/test_api.py

# Run fast unit tests only
poetry run pytest tests/unit/
```

Pull requests are expected to maintain or improve code coverage.

---

## Documentation Guidelines

- **Docstrings**: All public modules, classes, and functions must have Google-style docstrings explaining purpose, parameters, return types, and potential exceptions.
- **Architectural Changes**: Document major design decisions in relevant markdown artifacts or `docs/`.
- **API Changes**: Update Pydantic field descriptions in `src/serving/schemas.py` to ensure OpenAPI documentation at `/docs` reflects all changes.

---

## Submitting a Pull Request

Before opening a PR:
1. Ensure all tests pass: `poetry run pytest`.
2. Ensure linters pass: `poetry run ruff check .`.
3. Provide a clear PR description detailing:
   - What problem was solved or what feature was introduced.
   - Any model metric improvements (Brier score, PR-AUC, business cost).
   - Verification steps and test evidence.
