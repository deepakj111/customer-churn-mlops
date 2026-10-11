# Security Policy

The Customer Churn MLOps platform handles predictive models and customer features. We take the security and integrity of our data science services seriously.

---

## Supported Versions

Only the latest release branch receives active security updates and vulnerability patches.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

If you discover a potential security vulnerability within this project, please report it responsibly rather than opening a public issue.

### How to Report
1. Email your report to the repository maintainer or file a private security advisory on GitHub.
2. Please include:
   - Description of the vulnerability and affected component (`api`, `pipelines`, `feature store`).
   - Steps to reproduce or proof-of-concept payload.
   - Potential impact on model availability or data confidentiality.
3. We will acknowledge receipt within 48 hours and coordinate remediation.

---

## Platform Security Architecture & Controls

This repository implements multi-layered security safeguards aligned with OWASP API Security best practices:

### 1. Three-Tier Input Validation
- **Tier 1 (Pydantic v2)**: Rejects malformed JSON, out-of-bounds numerics, and unexpected categorical values before endpoint handlers execute (HTTP 422).
- **Tier 2 (Pandera DataFrames)**: Validates relational constraints and business invariants on converted tabular structures.
- **Tier 3 (Pipeline ColumnTransformer)**: Strictly constrains input column schemas to prevent pipeline injection attacks.

### 2. DoS & Abuse Protection
- **Sliding-Window Rate Limiting**: Enforces request rate limits per client IP (`RATE_LIMIT_PER_MINUTE=600`) returning HTTP 429 and `Retry-After` headers.
- **Resource Constraints**: Docker Compose services are bounded by strict CPU and memory limits to prevent host resource starvation.

### 3. Response Hardening & Information Leakage Prevention
- **OWASP HTTP Security Headers**: Every API response includes:
  - `X-Content-Type-Options: nosniff` (MIME sniffing prevention)
  - `X-Frame-Options: DENY` (Clickjacking mitigation)
  - `Cache-Control: no-store` (Prevents caching of sensitive predictions)
  - `Strict-Transport-Security: max-age=31536000; includeSubDomains` (Enforces HTTPS)
- **Centralized Exception Sanitization**: Global exception handlers trap unhandled errors and return structured `ErrorDetail` objects with unique `request_id` values, ensuring stack traces and internal file paths are never exposed to callers.

### 4. Container & Infrastructure Hardening
- **Read-Only Data Mounts**: Data volumes in Docker Compose are mounted with `:ro` (read-only) flags to prevent container tampering.
- **Isolated Network**: Services communicate over a dedicated internal bridge network without exposing MLflow or Feast stores publicly.
