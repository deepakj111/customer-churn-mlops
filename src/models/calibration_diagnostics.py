"""
Calibration Diagnostics — Quantitative Assessment of Probability Quality.

Traditional ML evaluation focuses on discrimination (ROC-AUC, PR-AUC):
can the model separate churners from non-churners?  Calibration answers
a different question: when the model says "70% churn probability", do
~70% of those customers actually churn?

Good calibration is a PREREQUISITE for:
    - Cost-optimal threshold tuning (threshold.py assumes probabilities
      are on a meaningful scale — if P=0.7 actually means P=0.4, the
      cost-optimal threshold is wrong)
    - Conformal prediction (conformal.py's coverage guarantee holds for
      any model, but prediction set sizes are tighter with calibrated
      probabilities)
    - Business risk tier assignment (schemas.py maps probability ranges
      to HIGH/MEDIUM/LOW — miscalibration makes these tiers lie)

Metrics implemented:
    ECE  — Expected Calibration Error (Naeini et al., AAAI 2015)
           Weighted average of per-bin |predicted - actual| gap.
           Lower is better. ECE < 0.05 is considered well-calibrated.

    MCE  — Maximum Calibration Error
           Worst-case bin miscalibration. Important for risk-sensitive
           applications where any probability range being wrong is dangerous.
           MCE < 0.10 is a reasonable target.

    Brier Decomposition — (Murphy, 1973)
           Decomposes Brier Score into three interpretable components:
               Reliability: How close P(y=1|p) is to p (lower = better calibrated)
               Resolution:  How much predictions vary from base rate (higher = better)
               Uncertainty: Entropy of the base rate (fixed for a given dataset)
           Brier = Reliability - Resolution + Uncertainty

    Reliability Diagram Data — For plotting calibration curves.

References:
    - Naeini, Cooper, Hauskrecht (2015), "Obtaining Well Calibrated
      Probabilities Using Bayesian Binning into Quantiles", AAAI
    - Murphy (1973), "A New Vector Partition of the Probability Score"
    - Guo et al. (2017), "On Calibration of Modern Neural Networks", ICML

Public API:
    compute_ece(y_true, y_prob, n_bins)                         → float
    compute_mce(y_true, y_prob, n_bins)                         → float
    compute_brier_decomposition(y_true, y_prob, n_bins)         → dict
    build_reliability_diagram_data(y_true, y_prob, n_bins)      → dict
    compute_calibration_report(y_true, y_prob_before, y_prob_after) → CalibrationReport
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Default number of bins for calibration metrics.
# 10 bins is standard (Naeini 2015, Guo 2017).  Fewer bins → less noise but
# coarser resolution; more bins → finer resolution but noisier estimates.
_DEFAULT_N_BINS = 10


# ---------------------------------------------------------------------------
# Core calibration metrics
# ---------------------------------------------------------------------------


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = _DEFAULT_N_BINS,
) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = Σ (n_b / N) × |acc(b) - conf(b)|

    where b indexes equal-width bins of predicted probability,
    acc(b) is the fraction of positives in bin b, and
    conf(b) is the mean predicted probability in bin b.

    Args:
        y_true:  Ground truth binary labels (0 or 1).
        y_prob:  Predicted probabilities for the positive class.
        n_bins:  Number of equal-width bins.

    Returns:
        ECE as a float in [0, 1]. Lower is better.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        # Include the lower boundary for the first bin
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        accuracy_in_bin = y_true[mask].mean()
        confidence_in_bin = y_prob[mask].mean()
        ece += (n_in_bin / len(y_true)) * abs(accuracy_in_bin - confidence_in_bin)

    return round(float(ece), 6)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = _DEFAULT_N_BINS,
) -> float:
    """
    Maximum Calibration Error (MCE).

    MCE = max_b |acc(b) - conf(b)|

    The worst-case bin miscalibration.  Important for risk-sensitive
    applications where even one probability range being wrong is
    unacceptable.

    Args:
        y_true:  Ground truth binary labels.
        y_prob:  Predicted probabilities.
        n_bins:  Number of equal-width bins.

    Returns:
        MCE as a float in [0, 1]. Lower is better.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_gap = 0.0

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        accuracy_in_bin = y_true[mask].mean()
        confidence_in_bin = y_prob[mask].mean()
        gap = abs(accuracy_in_bin - confidence_in_bin)
        max_gap = max(max_gap, gap)

    return round(float(max_gap), 6)


def compute_brier_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = _DEFAULT_N_BINS,
) -> dict[str, float]:
    """
    Decompose the Brier Score into reliability, resolution, uncertainty.

    Murphy (1973) decomposition:
        Brier = Reliability - Resolution + Uncertainty

    Components:
        Reliability: Σ (n_b/N) × (conf(b) - acc(b))²
            How close predicted probabilities are to observed frequencies.
            Lower = better calibrated. This is ECE² (weighted).

        Resolution: Σ (n_b/N) × (acc(b) - ȳ)²
            How much predictions deviate from the base rate.
            Higher = more discriminative. A constant predictor has
            resolution = 0.

        Uncertainty: ȳ × (1 - ȳ)
            Inherent entropy of the dataset. Fixed for a given y_true.
            Maximum at ȳ = 0.5 (most uncertain base rate).

    Args:
        y_true:  Ground truth binary labels.
        y_prob:  Predicted probabilities.
        n_bins:  Number of equal-width bins.

    Returns:
        Dict with keys: brier_score, reliability, resolution, uncertainty.
    """
    base_rate = y_true.mean()
    uncertainty = base_rate * (1.0 - base_rate)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        acc_b = y_true[mask].mean()
        conf_b = y_prob[mask].mean()
        weight = n_in_bin / n_total

        reliability += weight * (conf_b - acc_b) ** 2
        resolution += weight * (acc_b - base_rate) ** 2

    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "brier_score": round(brier, 6),
        "reliability": round(float(reliability), 6),
        "resolution": round(float(resolution), 6),
        "uncertainty": round(float(uncertainty), 6),
        "base_rate": round(float(base_rate), 4),
    }


def build_reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = _DEFAULT_N_BINS,
) -> dict[str, Any]:
    """
    Compute data for a reliability (calibration) diagram.

    Returns bin midpoints, observed frequencies, mean predicted
    probabilities, and bin counts — everything needed to plot the
    calibration curve and identify which probability ranges are
    over-confident vs. under-confident.

    A perfectly calibrated model's points lie on the y=x diagonal.
    Points above the diagonal → under-confident (actual > predicted).
    Points below the diagonal → over-confident (actual < predicted).

    Args:
        y_true:  Ground truth binary labels.
        y_prob:  Predicted probabilities.
        n_bins:  Number of equal-width bins.

    Returns:
        Dict with keys: bin_midpoints, fraction_positives,
        mean_predicted, bin_counts, n_total.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_midpoints = []
    fraction_positives = []
    mean_predicted = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        bin_midpoints.append(round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 3))
        fraction_positives.append(round(float(y_true[mask].mean()), 4))
        mean_predicted.append(round(float(y_prob[mask].mean()), 4))
        bin_counts.append(int(n_in_bin))

    return {
        "bin_midpoints": bin_midpoints,
        "fraction_positives": fraction_positives,
        "mean_predicted": mean_predicted,
        "bin_counts": bin_counts,
        "n_total": len(y_true),
    }


# ---------------------------------------------------------------------------
# Calibration Report — Pre vs. Post calibration comparison
# ---------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    """Complete calibration assessment comparing pre/post calibration."""

    # Pre-calibration metrics
    pre_ece: float
    pre_mce: float
    pre_brier: dict[str, float]
    pre_reliability_diagram: dict[str, Any]

    # Post-calibration metrics
    post_ece: float
    post_mce: float
    post_brier: dict[str, float]
    post_reliability_diagram: dict[str, Any]

    # Improvement deltas (negative = improved)
    ece_improvement: float = field(init=False)
    mce_improvement: float = field(init=False)
    brier_improvement: float = field(init=False)

    def __post_init__(self) -> None:
        self.ece_improvement = round(self.post_ece - self.pre_ece, 6)
        self.mce_improvement = round(self.post_mce - self.pre_mce, 6)
        self.brier_improvement = round(
            self.post_brier["brier_score"] - self.pre_brier["brier_score"], 6
        )

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a dict suitable for MLflow logging."""
        result: dict[str, Any] = {
            "pre_calibration_ece": self.pre_ece,
            "pre_calibration_mce": self.pre_mce,
            "pre_calibration_brier": self.pre_brier["brier_score"],
            "pre_calibration_reliability": self.pre_brier["reliability"],
            "pre_calibration_resolution": self.pre_brier["resolution"],
            "post_calibration_ece": self.post_ece,
            "post_calibration_mce": self.post_mce,
            "post_calibration_brier": self.post_brier["brier_score"],
            "post_calibration_reliability": self.post_brier["reliability"],
            "post_calibration_resolution": self.post_brier["resolution"],
            "ece_improvement": self.ece_improvement,
            "mce_improvement": self.mce_improvement,
            "brier_improvement": self.brier_improvement,
        }
        return result

    def summary(self) -> str:
        """Human-readable calibration report."""
        lines = [
            "=" * 60,
            "  CALIBRATION DIAGNOSTICS REPORT",
            "=" * 60,
            "",
            "  Metric               Pre-Cal    Post-Cal   Δ (neg=better)",
            "  " + "-" * 56,
            (
                f"  ECE                  {self.pre_ece:.4f}     "
                f"{self.post_ece:.4f}     {self.ece_improvement:+.4f}"
            ),
            (
                f"  MCE                  {self.pre_mce:.4f}     "
                f"{self.post_mce:.4f}     {self.mce_improvement:+.4f}"
            ),
            (
                f"  Brier Score          {self.pre_brier['brier_score']:.4f}     "
                f"{self.post_brier['brier_score']:.4f}     "
                f"{self.brier_improvement:+.4f}"
            ),
            "",
            "  Brier Decomposition (Post-Calibration):",
            (
                f"    Reliability  = {self.post_brier['reliability']:.6f}  "
                "(lower = better calibrated)"
            ),
            (
                f"    Resolution   = {self.post_brier['resolution']:.6f}  "
                "(higher = more discriminative)"
            ),
            (
                f"    Uncertainty  = {self.post_brier['uncertainty']:.6f}  "
                "(fixed by dataset)"
            ),
            "=" * 60,
        ]
        return "\n".join(lines)


def compute_calibration_report(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    n_bins: int = _DEFAULT_N_BINS,
) -> CalibrationReport:
    """
    Compare calibration quality before and after isotonic calibration.

    This is the main entry point. Call after CalibratedClassifierCV
    to quantify how much calibration actually improved.

    Args:
        y_true:        Ground truth binary labels.
        y_prob_before: Probabilities from the raw (uncalibrated) model.
        y_prob_after:  Probabilities from the calibrated model.
        n_bins:        Number of bins for all calibration metrics.

    Returns:
        CalibrationReport with full pre/post comparison and improvement deltas.
    """
    logger.info("Computing calibration diagnostics (pre vs. post calibration)...")

    report = CalibrationReport(
        pre_ece=compute_ece(y_true, y_prob_before, n_bins),
        pre_mce=compute_mce(y_true, y_prob_before, n_bins),
        pre_brier=compute_brier_decomposition(y_true, y_prob_before, n_bins),
        pre_reliability_diagram=build_reliability_diagram_data(
            y_true, y_prob_before, n_bins
        ),
        post_ece=compute_ece(y_true, y_prob_after, n_bins),
        post_mce=compute_mce(y_true, y_prob_after, n_bins),
        post_brier=compute_brier_decomposition(y_true, y_prob_after, n_bins),
        post_reliability_diagram=build_reliability_diagram_data(
            y_true, y_prob_after, n_bins
        ),
    )

    logger.info(
        "Calibration diagnostics — ECE: %.4f → %.4f (%+.4f), "
        "Brier: %.4f → %.4f (%+.4f)",
        report.pre_ece,
        report.post_ece,
        report.ece_improvement,
        report.pre_brier["brier_score"],
        report.post_brier["brier_score"],
        report.brier_improvement,
    )

    return report
