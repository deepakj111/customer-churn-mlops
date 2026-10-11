# Changelog

All notable changes to the Customer Churn MLOps platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-11

### Added
- **API Versioning & Routing**: Prefixed production endpoints under `/api/v1/` (`/api/v1/predict`, `/api/v1/predict/batch`, `/api/v1/model/info`, `/api/v1/uplift/*`) while maintaining full backward-compatible root routes.
- **Asynchronous Batch Inference**: Added non-blocking batch submission endpoint (`POST /api/v1/predict/batch/async`) returning HTTP 202 Accepted and job polling endpoint (`GET /api/v1/predict/batch/jobs/{job_id}`).
- **Kubernetes Readiness Probes**: Added dedicated `/readiness` endpoint distinguishing container liveness (`/health`) from model serving readiness.
- **OWASP API Security Hardening**: Injected standard security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Cache-Control: no-store`, `Strict-Transport-Security`).
- **Sliding-Window Rate Limiting**: Implemented configurable in-memory rate limiting middleware with `X-RateLimit-*` and `Retry-After` headers.
- **Structured JSON Logging**: Implemented `_JSONFormatter` selectable via `LOG_FORMAT=json` for cloud log aggregators (Datadog, CloudWatch, ELK).
- **Container Hardening**: Added CPU and memory resource limits/reservations and readiness health checks across all services in `docker-compose.yml`.
- **Engineering Hygiene**: Added `CONTRIBUTING.md`, `SECURITY.md`, and formalized release changelog.

### Fixed
- **Cost Matrix Inconsistency**: Aligned `false_positive_cost` in `configs/model_config.yaml` to $50.00, achieving consistency across model configuration, model card, and documentation.
- **CLI Subprocess Parameter Bug**: Fixed parameter mapping in `main.py` when invoking `run_retraining_pipeline(force_retrain=args.force)`.
- **Drift Script CLI Parsing**: Refactored `scripts/generate_drift_report.py` to accept arguments directly without mutating global `sys.argv`.
- **Repository Hygiene**: Removed debug artifacts (`repo_code*`) from `.gitignore` and eliminated legacy comment noise in configuration files.

---

## [0.9.0] - 2026-09-15

### Added
- **Causal AI & Uplift Modeling**: Implemented EconML CATE T-Learner estimating individual treatment effects to segment customers into Persuadables, Sure Things, Lost Causes, and Sleeping Dogs.
- **Conformal Prediction Sets**: Added non-conformity scoring delivering guaranteed confidence coverage bounds for high-stakes churn predictions.
- **Champion-Challenger Automated Retraining**: Automated Prefect retraining flow comparing challengers using Nadeau-Bengio corrected resampled t-tests.
- **Fairness Audits**: Incorporated Demographic Parity and Equalized Odds disparity evaluations across protected demographic attributes.
- **Feature Store Integration**: Feast online and offline feature store definitions with dynamic real-time entity retrieval.
