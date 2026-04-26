# ============================================================================
# Multi-stage Dockerfile for the Customer Churn Prediction API.
#
# Stage 1 (builder):
#   - Installs Poetry and all project dependencies.
#   - Exports them to a requirements.txt for pip install in the runtime stage.
#   - This avoids shipping Poetry itself in the final image.
#
# Stage 2 (runtime):
#   - Slim Python image (~150MB vs ~900MB for full).
#   - Non-root user for security best practices.
#   - Built-in health check for container orchestrators.
#   - Only production dependencies — no dev tools, no test suites.
#
# Build:
#   docker build -f docker/Dockerfile -t churn-api:latest .
#
# Run:
#   docker run -p 8000:8000 --env-file .env churn-api:latest
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — export dependencies from Poetry
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install Poetry (locked version for reproducibility)
ENV POETRY_VERSION=2.3.3
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}" poetry-plugin-export

# Copy only dependency files first — maximises Docker layer caching.
# Source code changes don't bust this expensive layer.
COPY pyproject.toml poetry.lock ./

# Export production dependencies to a plain requirements.txt.
# --without dev: exclude pytest, black, jupyter, etc.
# --no-interaction: don't prompt for anything.
RUN poetry export \
    --format requirements.txt \
    --output /build/requirements.txt \
    --no-interaction

# ---------------------------------------------------------------------------
# Stage 2: Runtime — minimal production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Metadata labels following OCI conventions
LABEL maintainer="Deepak Jangra" \
      description="Customer Churn Prediction API — FastAPI + LightGBM" \
      version="0.1.0"

# Prevent Python from writing .pyc files and buffering stdout/stderr.
# Unbuffered output is essential for Docker log drivers to capture logs in
# real-time instead of waiting for Python's internal buffer to flush.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system-level dependencies required by LightGBM and numpy.
# libgomp1: OpenMP runtime for LightGBM parallel inference.
# These are runtime-only — no compilers or build tools shipped.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies from the exported requirements.txt.
# --no-cache-dir: don't store pip's download cache inside the image.
COPY --from=builder /build/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code.
# The .dockerignore excludes .venv, .git, notebooks, tests, etc.
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/raw/ ./data/raw/
COPY main.py ./

# Create a non-root user for security.
# Running as root inside containers is a well-known security anti-pattern.
# If the container is compromised, the attacker has root privileges on the
# container's filesystem. A non-root user limits the blast radius.
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Expose the API port (documentation — doesn't actually publish)
EXPOSE 8000

# Health check — container orchestrators (Docker Swarm, ECS, k8s) use this
# to determine if the container is alive and ready to receive traffic.
# The /health endpoint returns 200 even if the model isn't loaded yet
# (it reports model_loaded=false), which is intentional — liveness != readiness.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server with Uvicorn.
# --workers 1: single worker per container (scale via container replicas).
# --host 0.0.0.0: listen on all interfaces (required inside Docker).
# --port 8000: matches the EXPOSE directive and health check.
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
