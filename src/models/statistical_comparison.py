"""
Statistical Model Comparison — Rigorous Hypothesis Testing for Benchmarks.

When comparing two models on cross-validation results, the raw difference
in mean scores is NOT sufficient to conclude that one model is better.
CV folds share training data, making fold scores non-independent — a
standard t-test produces grossly anti-conservative p-values.

This module implements the corrected repeated k-fold t-test (Nadeau &
Bengio, 2003), which accounts for this non-independence and produces
valid p-values for CV model comparison.

Why this matters:
    A 0.02 AUC difference between LightGBM and FT-Transformer might be:
    (a) A real, reproducible performance gap, OR
    (b) Random noise from the particular train/test splits.
    Without a statistical test, you cannot make this distinction.
    Reporting the difference as "significant" without testing is a
    common methodological error in applied ML.

Metrics provided:
    Corrected t-test — Accounts for non-independence of CV folds.
    Cohen's d        — Effect size: practical significance, not just
                       statistical significance. d > 0.8 is "large".
    Confidence Interval — 95% CI for the true performance difference.

References:
    - Nadeau & Bengio (2003), "Inference for the Generalization Error",
      Machine Learning, 52(3), pp. 239-281
    - Bouckaert & Frank (2004), "Evaluating the Replicability of
      Significance Tests for Comparing Learning Algorithms", PAKDD
    - Dietterich (1998), "Approximate Statistical Tests for Comparing
      Supervised Classification Learning Algorithms", Neural Computation

Public API:
    corrected_cv_ttest(scores_a, scores_b, n_train, n_test) → TestResult
    cohens_d(scores_a, scores_b)                            → float
    compare_models(results_dict, n_train, n_test)           → ComparisonReport
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TestResult:
    """Result of a single pairwise statistical test."""

    model_a: str
    model_b: str
    metric_name: str
    mean_a: float
    mean_b: float
    mean_diff: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    cohens_d: float
    is_significant: bool  # at alpha = 0.05
    effect_interpretation: str  # "negligible", "small", "medium", "large"


def corrected_cv_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_train: int,
    n_test: int,
    alpha: float = 0.05,
    model_a: str = "Model A",
    model_b: str = "Model B",
    metric_name: str = "score",
) -> TestResult:
    """
    Corrected repeated k-fold cross-validation t-test.

    Standard paired t-test assumes independence between pairs, but CV
    folds share training data — making scores correlated.  Nadeau &
    Bengio (2003) correct the variance estimate:

        σ²_corrected = (1/k + n_test/n_train) × σ²_diff

    where k is the number of folds, n_test and n_train are the fold
    sizes, and σ²_diff is the variance of the per-fold differences.

    The correction factor (1/k + n_test/n_train) inflates the variance
    to account for the overlap — producing wider confidence intervals
    and more conservative (honest) p-values.

    Args:
        scores_a:    Per-fold scores for model A (length k).
        scores_b:    Per-fold scores for model B (length k).
        n_train:     Number of training samples per fold.
        n_test:      Number of test samples per fold.
        alpha:       Significance level (default 0.05).
        model_a:     Name of model A (for reporting).
        model_b:     Name of model B (for reporting).
        metric_name: Name of the metric being compared.

    Returns:
        TestResult with t-statistic, corrected p-value, 95% CI,
        and effect size interpretation.
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    k = len(scores_a)

    if len(scores_a) != len(scores_b):
        raise ValueError(
            "Score arrays must have equal length. "
            f"Got {len(scores_a)} and {len(scores_b)}."
        )

    # Per-fold differences
    diffs = scores_a - scores_b
    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1))

    # Nadeau-Bengio correction factor
    correction = (1.0 / k) + (n_test / n_train)
    corrected_var = correction * var_diff
    corrected_se = np.sqrt(corrected_var) if corrected_var > 0 else 1e-10

    # t-statistic and p-value (two-tailed)
    t_stat = mean_diff / corrected_se
    df = k - 1
    p_value = float(2 * stats.t.sf(abs(t_stat), df))

    # 95% Confidence interval for the true difference
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    ci_lower = mean_diff - t_crit * corrected_se
    ci_upper = mean_diff + t_crit * corrected_se

    # Effect size: Cohen's d
    d = _cohens_d(scores_a, scores_b)

    # Interpret effect size (Cohen's conventions)
    if abs(d) < 0.2:
        effect_interp = "negligible"
    elif abs(d) < 0.5:
        effect_interp = "small"
    elif abs(d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    return TestResult(
        model_a=model_a,
        model_b=model_b,
        metric_name=metric_name,
        mean_a=round(float(np.mean(scores_a)), 4),
        mean_b=round(float(np.mean(scores_b)), 4),
        mean_diff=round(mean_diff, 4),
        t_statistic=round(float(t_stat), 4),
        p_value=round(float(p_value), 6),
        ci_lower=round(float(ci_lower), 4),
        ci_upper=round(float(ci_upper), 4),
        cohens_d=round(d, 4),
        is_significant=p_value < alpha,
        effect_interpretation=effect_interp,
    )


def _cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    d = mean(diff) / std(diff)

    Interpretation (Cohen, 1988):
        |d| < 0.2  : negligible
        |d| < 0.5  : small
        |d| < 0.8  : medium
        |d| >= 0.8 : large

    Returns:
        Cohen's d as a float. Positive means A > B.
    """
    diffs = scores_a - scores_b
    std_diff = float(np.std(diffs, ddof=1))
    if std_diff == 0:
        return 0.0
    return float(np.mean(diffs) / std_diff)


# ---------------------------------------------------------------------------
# Multi-model comparison report
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """Pairwise statistical comparison of all models."""

    pairwise_tests: list[TestResult]
    n_comparisons: int
    bonferroni_alpha: float  # Corrected significance level for multiple testing
    summary_table: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict for JSON export."""
        result: dict[str, Any] = {
            "n_comparisons": self.n_comparisons,
            "bonferroni_alpha": self.bonferroni_alpha,
        }
        for test in self.pairwise_tests:
            key = f"{test.model_a}_vs_{test.model_b}_{test.metric_name}"
            result[f"{key}_p_value"] = test.p_value
            result[f"{key}_cohens_d"] = test.cohens_d
            result[f"{key}_significant"] = test.is_significant
            result[f"{key}_ci"] = [test.ci_lower, test.ci_upper]
        return result

    def summary(self) -> str:
        """Human-readable comparison report."""
        lines = [
            "=" * 75,
            "  STATISTICAL MODEL COMPARISON (Nadeau-Bengio Corrected t-test)",
            "=" * 75,
            f"  Comparisons: {self.n_comparisons}, "
            f"Bonferroni α: {self.bonferroni_alpha:.4f}",
            "",
        ]

        for test in self.pairwise_tests:
            sig = "SIGNIFICANT" if test.is_significant else "not significant"
            lines.append(f"  {test.model_a} vs {test.model_b} [{test.metric_name}]")
            lines.append(
                f"    Means: {test.mean_a:.4f} vs {test.mean_b:.4f} "
                f"(Δ = {test.mean_diff:+.4f})"
            )
            lines.append(f"    95% CI: [{test.ci_lower:+.4f}, {test.ci_upper:+.4f}]")
            lines.append(f"    p = {test.p_value:.6f} ({sig})")
            lines.append(
                f"    Cohen's d = {test.cohens_d:+.4f} ({test.effect_interpretation})"
            )
            lines.append("")

        lines.append("=" * 75)
        return "\n".join(lines)


def compare_models(
    fold_scores: dict[str, np.ndarray],
    n_train: int,
    n_test: int,
    metric_name: str = "ROC-AUC",
    alpha: float = 0.05,
) -> ComparisonReport:
    """
    Perform all pairwise model comparisons with Bonferroni correction.

    Args:
        fold_scores:  Dict mapping model_name → array of per-fold scores.
        n_train:      Number of training samples per fold.
        n_test:       Number of test samples per fold.
        metric_name:  Name of the metric (for reporting).
        alpha:        Family-wise significance level (before Bonferroni).

    Returns:
        ComparisonReport with all pairwise tests and multiple testing
        correction.
    """
    model_names = list(fold_scores.keys())
    pairs = list(combinations(model_names, 2))
    n_comparisons = len(pairs)

    # Bonferroni correction: divide alpha by number of tests
    bonferroni_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha

    tests = []
    for model_a, model_b in pairs:
        result = corrected_cv_ttest(
            scores_a=fold_scores[model_a],
            scores_b=fold_scores[model_b],
            n_train=n_train,
            n_test=n_test,
            alpha=bonferroni_alpha,
            model_a=model_a,
            model_b=model_b,
            metric_name=metric_name,
        )
        tests.append(result)

        logger.info(
            "  %s vs %s [%s]: Δ = %+.4f, p = %.6f, d = %+.4f (%s)",
            model_a,
            model_b,
            metric_name,
            result.mean_diff,
            result.p_value,
            result.cohens_d,
            "SIG" if result.is_significant else "n.s.",
        )

    report = ComparisonReport(
        pairwise_tests=tests,
        n_comparisons=n_comparisons,
        bonferroni_alpha=round(bonferroni_alpha, 6),
    )

    return report
