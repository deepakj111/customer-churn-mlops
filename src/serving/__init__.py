"""
Serving module — FastAPI prediction API.

Exports the FastAPI app instance for uvicorn:
    uvicorn src.serving.api:app --reload
"""

from src.serving.api import app  # noqa: F401

__all__ = ["app"]
