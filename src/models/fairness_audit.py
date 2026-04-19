"""
Fairness / Bias Audit — Responsible AI Assessment for Churn Prediction.

A model that predicts churn accurately on average may still perform
unfairly across demographic groups.  This module audits the model for
disparate impact and differential accuracy across protected attributes.

Why this matters for churn prediction:
    The Telco dataset includes gender and SeniorCitizen as features.
    If the model systematically flags more seniors for churn outreach
    (False Positives) while missing younger churners (False Negatives),
    the retention team's campaigns are inequitably distributed — and
    potentially violating anti-discrimination regulations (ECOA, GDPR).

Metrics implemented:
    Demographic Parity Ratio — P(ŷ=1|group=A) / P(ŷ=1|group=B)
        Measures whether the model flags each group at equal rates.
        Ratio = 1.0 means perfect parity. Values in [0.8, 1.25] are
        typically considered acceptable (4/5ths rule, EEOC guidelines).

    Equalized Odds — TPR and FPR disparity across groups
        Measures whether the model is equally accurate for each group.
        Small TPR disparity = model catches churners equally well.
        Small FPR disparity = model doesn't disproportionately flag
        non-churners in one group.

    Group-Level Calibration — ECE computed per group
        Measures whether the model's probability estimates are
        equally trustworthy across groups.

    Group-Level Performance — per-group precision, recall, F1, AUC

References:
    - Hardt, Price, Srebro (2016), "Equality of Opportunity in
      Supervised Learning", NeurIPS
    - Chouldechova (2017), "Fair Prediction with Disparate Impact"
    - Barocas, Hardt, Narayanan (2019), "Fairness and Machine
      Learning: Limitations and Opportunities"

Public API:
    compute_group_metrics(y_true, y_pred, y_prob, groups)  → dict per group
    compute_demographic_parity(y_pred, groups)             → DemographicParityResult
    compute_equalized_odds(y_true, y_pred, groups)         → EqualizedOddsResult
    run_fairness_audit(y_true, y_pred, y_prob, sensitive_df) → FairnessReport
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.models.calibration_diagnostics import compute_ece
from src.utils.logging import get_logger

logger = get_logger(__name__)

# 4/5ths rule threshold — EEOC guidelines for disparate impact.
# If the selection rate ratio falls below 0.8, there is a prima facie
# case for adverse impact.
_FOUR_FIFTHS_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Group-level metrics
# ---------------------------------------------------------------------------


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """
    Compute per-group classification metrics.

    For each distinct value in `groups`, computes precision, recall,
    F1, AUC, positive rate, ECE, and sample size.

    Args:
        y_true:  Ground truth labels (0 or 1).
        y_pred:  Binary predictions at the chosen threshold.
        y_prob:  Predicted probabilities.
        groups:  Group membership array (same length as y_true).

    Returns:
        Dict mapping group_name → metrics dict.
    """
    unique_groups = np.unique(groups)
    results: dict[str, dict[str, Any]] = {}

    for group in unique_groups:
        mask = groups == group
        n = int(mask.sum())

        if n < 10:
            logger.warning(
                "Group '%s' has only %d samples — metrics may be unreliable.",
                group,
                n,
            )

        yt = y_true[mask]
        yp = y_pred[mask]
        yprob = y_prob[mask]

        # AUC requires both classes present
        try:
            auc = round(float(roc_auc_score(yt, yprob)), 4)
        except ValueError:
            auc = float("nan")

        results[str(group)] = {
            "n_samples": n,
            "base_rate": round(float(yt.mean()), 4),
            "positive_rate": round(float(yp.mean()), 4),
            "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
            "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
            "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
            "roc_auc": auc,
            "ece": compute_ece(yt, yprob, n_bins=10),
        }

    return results


# ---------------------------------------------------------------------------
# Demographic Parity
# ---------------------------------------------------------------------------


@dataclass
class DemographicParityResult:
    """Result of demographic parity analysis for a single attribute."""

    attribute_name: str
    group_positive_rates: dict[str, float]
    parity_ratio: float
    passes_four_fifths: bool
    min_rate_group: str
    max_rate_group: str


def compute_demographic_parity(
    y_pred: np.ndarray,
    groups: np.ndarray,
    attribute_name: str = "unknown",
) -> DemographicParityResult:
    """
    Compute Demographic Parity Ratio across groups.

    Demographic Parity requires: P(ŷ=1|group=A) = P(ŷ=1|group=B)
    The ratio measures how close the positive rates are.

    Ratio = min_group_rate / max_group_rate

    Args:
        y_pred:          Binary predictions.
        groups:          Group membership array.
        attribute_name:  Name of the sensitive attribute (for reporting).

    Returns:
        DemographicParityResult with ratio and pass/fail assessment.
    """
    unique_groups = np.unique(groups)
    rates: dict[str, float] = {}

    for group in unique_groups:
        mask = groups == group
        rates[str(group)] = round(float(y_pred[mask].mean()), 4)

    if not rates or max(rates.values()) == 0:
        return DemographicParityResult(
            attribute_name=attribute_name,
            group_positive_rates=rates,
            parity_ratio=0.0,
            passes_four_fifths=False,
            min_rate_group="N/A",
            max_rate_group="N/A",
        )

    min_group = min(rates, key=rates.get)  # type: ignore[arg-type]
    max_group = max(rates, key=rates.get)  # type: ignore[arg-type]
    ratio = (
        round(rates[min_group] / rates[max_group], 4) if rates[max_group] > 0 else 0.0
    )

    return DemographicParityResult(
        attribute_name=attribute_name,
        group_positive_rates=rates,
        parity_ratio=ratio,
        passes_four_fifths=ratio >= _FOUR_FIFTHS_THRESHOLD,
        min_rate_group=min_group,
        max_rate_group=max_group,
    )


# ---------------------------------------------------------------------------
# Equalized Odds
# ---------------------------------------------------------------------------


@dataclass
class EqualizedOddsResult:
    """Result of equalized odds analysis for a single attribute."""

    attribute_name: str
    group_tpr: dict[str, float]
    group_fpr: dict[str, float]
    tpr_disparity: float  # max - min TPR across groups
    fpr_disparity: float  # max - min FPR across groups
    satisfies_equalized_odds: bool  # both disparities < 0.10


def compute_equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    attribute_name: str = "unknown",
    disparity_threshold: float = 0.10,
) -> EqualizedOddsResult:
    """
    Compute Equalized Odds: TPR and FPR parity across groups.

    Equalized Odds (Hardt et al., 2016) requires:
        P(ŷ=1|Y=1, group=A) = P(ŷ=1|Y=1, group=B)   [equal TPR]
        P(ŷ=1|Y=0, group=A) = P(ŷ=1|Y=0, group=B)   [equal FPR]

    Args:
        y_true:  Ground truth labels.
        y_pred:  Binary predictions.
        groups:  Group membership array.
        attribute_name:  Name of the sensitive attribute.
        disparity_threshold:  Maximum acceptable TPR/FPR gap.

    Returns:
        EqualizedOddsResult with per-group TPR/FPR and disparity measures.
    """
    unique_groups = np.unique(groups)
    tpr_dict: dict[str, float] = {}
    fpr_dict: dict[str, float] = {}

    for group in unique_groups:
        mask = groups == group
        yt = y_true[mask]
        yp = y_pred[mask]

        # TPR = recall = TP / (TP + FN)
        positives = yt == 1
        if positives.sum() > 0:
            tpr_dict[str(group)] = round(float(yp[positives].mean()), 4)
        else:
            tpr_dict[str(group)] = float("nan")

        # FPR = FP / (FP + TN)
        negatives = yt == 0
        if negatives.sum() > 0:
            fpr_dict[str(group)] = round(float(yp[negatives].mean()), 4)
        else:
            fpr_dict[str(group)] = float("nan")

    tpr_vals = [v for v in tpr_dict.values() if not np.isnan(v)]
    fpr_vals = [v for v in fpr_dict.values() if not np.isnan(v)]

    tpr_disparity = round(max(tpr_vals) - min(tpr_vals), 4) if tpr_vals else 0.0
    fpr_disparity = round(max(fpr_vals) - min(fpr_vals), 4) if fpr_vals else 0.0

    return EqualizedOddsResult(
        attribute_name=attribute_name,
        group_tpr=tpr_dict,
        group_fpr=fpr_dict,
        tpr_disparity=tpr_disparity,
        fpr_disparity=fpr_disparity,
        satisfies_equalized_odds=(
            tpr_disparity < disparity_threshold and fpr_disparity < disparity_threshold
        ),
    )


# ---------------------------------------------------------------------------
# Full Fairness Report
# ---------------------------------------------------------------------------


@dataclass
class FairnessReport:
    """Complete fairness audit across all sensitive attributes."""

    sensitive_attributes: list[str]
    group_metrics: dict[str, dict[str, dict[str, Any]]]  # attr → group → metrics
    demographic_parity: dict[str, DemographicParityResult]  # attr → result
    equalized_odds: dict[str, EqualizedOddsResult]  # attr → result
    overall_fair: bool = field(init=False)

    def __post_init__(self) -> None:
        """A model passes the overall fairness check if ALL attributes pass."""
        dp_pass = all(r.passes_four_fifths for r in self.demographic_parity.values())
        eo_pass = all(r.satisfies_equalized_odds for r in self.equalized_odds.values())
        self.overall_fair = dp_pass and eo_pass

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a dict suitable for MLflow logging."""
        result: dict[str, Any] = {"overall_fair": self.overall_fair}

        for attr, dp in self.demographic_parity.items():
            result[f"fairness_{attr}_dp_ratio"] = dp.parity_ratio
            result[f"fairness_{attr}_dp_passes"] = dp.passes_four_fifths

        for attr, eo in self.equalized_odds.items():
            result[f"fairness_{attr}_tpr_disparity"] = eo.tpr_disparity
            result[f"fairness_{attr}_fpr_disparity"] = eo.fpr_disparity
            result[f"fairness_{attr}_eo_passes"] = eo.satisfies_equalized_odds

        return result

    def summary(self) -> str:
        """Human-readable fairness audit report."""
        lines = [
            "=" * 65,
            "  FAIRNESS / BIAS AUDIT REPORT",
            "=" * 65,
        ]

        for attr in self.sensitive_attributes:
            dp = self.demographic_parity[attr]
            eo = self.equalized_odds[attr]
            flag_dp = "✅" if dp.passes_four_fifths else "⚠️"
            flag_eo = "✅" if eo.satisfies_equalized_odds else "⚠️"

            lines.append(f"\n  Attribute: {attr}")
            lines.append("  " + "-" * 55)
            lines.append(
                f"  Demographic Parity Ratio : {dp.parity_ratio:.4f}  {flag_dp}"
            )
            lines.append(f"    Positive rates: {dp.group_positive_rates}")
            lines.append(f"  Equalized Odds:  {flag_eo}")
            lines.append(
                f"    TPR disparity: {eo.tpr_disparity:.4f}  (max gap across groups)"
            )
            lines.append(f"    FPR disparity: {eo.fpr_disparity:.4f}")
            lines.append(f"    Group TPRs: {eo.group_tpr}")
            lines.append(f"    Group FPRs: {eo.group_fpr}")

        overall = "PASS ✅" if self.overall_fair else "ISSUES DETECTED ⚠️"
        lines.append(f"\n  Overall Fairness: {overall}")
        lines.append("=" * 65)
        return "\n".join(lines)


def run_fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive_features: dict[str, np.ndarray],
) -> FairnessReport:
    """
    Run a comprehensive fairness audit across all sensitive attributes.

    Args:
        y_true:  Ground truth labels.
        y_pred:  Binary predictions at the chosen threshold.
        y_prob:  Predicted probabilities.
        sensitive_features:  Dict mapping attribute_name → group_array.
            Example: {"gender": gender_array, "SeniorCitizen": senior_array}

    Returns:
        FairnessReport with per-attribute demographic parity, equalized
        odds, and group-level metrics.
    """
    logger.info(
        "Running fairness audit on %d attributes: %s",
        len(sensitive_features),
        list(sensitive_features.keys()),
    )

    group_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    dp_results: dict[str, DemographicParityResult] = {}
    eo_results: dict[str, EqualizedOddsResult] = {}

    for attr_name, groups in sensitive_features.items():
        group_metrics[attr_name] = compute_group_metrics(y_true, y_pred, y_prob, groups)
        dp_results[attr_name] = compute_demographic_parity(
            y_pred, groups, attribute_name=attr_name
        )
        eo_results[attr_name] = compute_equalized_odds(
            y_true, y_pred, groups, attribute_name=attr_name
        )

        logger.info(
            "  %s — DP ratio: %.4f (%s), TPR gap: %.4f, FPR gap: %.4f (%s)",
            attr_name,
            dp_results[attr_name].parity_ratio,
            "PASS" if dp_results[attr_name].passes_four_fifths else "FAIL",
            eo_results[attr_name].tpr_disparity,
            eo_results[attr_name].fpr_disparity,
            "PASS" if eo_results[attr_name].satisfies_equalized_odds else "FAIL",
        )

    report = FairnessReport(
        sensitive_attributes=list(sensitive_features.keys()),
        group_metrics=group_metrics,
        demographic_parity=dp_results,
        equalized_odds=eo_results,
    )

    logger.info(
        "Fairness audit complete — overall: %s",
        "FAIR" if report.overall_fair else "ISSUES",
    )
    return report
