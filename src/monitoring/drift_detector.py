"""Drift detection (IPP-008).

The interviewer's exact pain: a model that shines offline but degrades on 2M live
invoices as templates, suppliers and layouts shift. We can't wait for accuracy to
crater — we watch *leading indicators* that move before quality does:

  1. Confidence drift — has the score distribution shifted vs. a baseline window?
     Measured with PSI (Population Stability Index), the standard for this.
  2. New-supplier rate — a surge of unseen suppliers is exactly when extraction
     quality falls off, and it's observable with zero labels.

Both are label-free, so they run continuously in production and trigger a
retraining/investigation alert long before the correction rate spikes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# PSI interpretation (industry convention):
#   < 0.1 stable · 0.1–0.25 moderate shift · > 0.25 significant shift
PSI_ALERT_THRESHOLD = 0.25
NEW_SUPPLIER_ALERT_RATE = 0.30


@dataclass
class DriftReport:
    confidence_psi: float
    new_supplier_rate: float
    drift_detected: bool
    reasons: list[str]


def population_stability_index(
    baseline: list[float], current: list[float], n_bins: int = 10
) -> float:
    """PSI between two distributions of scores in [0, 1].

    PSI = sum over bins of (curr% - base%) * ln(curr% / base%). Empty bins are
    floored with a small epsilon so the log stays finite.
    """
    if not baseline or not current:
        return 0.0

    eps = 1e-6
    edges = [i / n_bins for i in range(n_bins + 1)]
    edges[-1] = 1.0 + eps  # include 1.0 in the last bin

    def dist(values: list[float]) -> list[float]:
        counts = [0] * n_bins
        for v in values:
            for i in range(n_bins):
                if edges[i] <= v < edges[i + 1]:
                    counts[i] += 1
                    break
        total = len(values)
        return [max(c / total, eps) for c in counts]

    base_d, curr_d = dist(baseline), dist(current)
    return round(sum((c - b) * math.log(c / b) for b, c in zip(base_d, curr_d)), 4)


def detect_drift(
    baseline_confidences: list[float],
    current_confidences: list[float],
    baseline_suppliers: set[str],
    current_suppliers: list[str],
) -> DriftReport:
    psi = population_stability_index(baseline_confidences, current_confidences)

    unseen = [s for s in current_suppliers if s not in baseline_suppliers]
    new_rate = round(len(unseen) / len(current_suppliers), 3) if current_suppliers else 0.0

    reasons: list[str] = []
    if psi > PSI_ALERT_THRESHOLD:
        reasons.append(f"confidence_psi {psi} > {PSI_ALERT_THRESHOLD}")
    if new_rate > NEW_SUPPLIER_ALERT_RATE:
        reasons.append(f"new_supplier_rate {new_rate} > {NEW_SUPPLIER_ALERT_RATE}")

    return DriftReport(
        confidence_psi=psi,
        new_supplier_rate=new_rate,
        drift_detected=bool(reasons),
        reasons=reasons,
    )
