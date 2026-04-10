"""
Monitoring module — data drift detection and reference data management.

Exports:
    DriftDetector       — detects feature and prediction drift
    build_reference     — creates training data snapshots for comparison
    load_reference      — loads reference snapshots from disk
"""

from src.monitoring.drift_detector import DriftDetector, DriftReport  # noqa: F401
from src.monitoring.reference_builder import (  # noqa: F401
    build_reference,
    load_reference,
)

__all__ = [
    "DriftDetector",
    "DriftReport",
    "build_reference",
    "load_reference",
]
