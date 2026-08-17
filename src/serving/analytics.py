"""Business analytics (IPP-008).

Computes the metrics that actually matter for an invoice-processing product —
straight from the database, so they always reflect real state. These are the
"are we an ML *product*, not just a model" numbers the interviewer emphasised:

  * automatic_processing_rate — the headline: what share booked with no human?
  * correction_rate           — how often does a human have to fix us?
  * field_error_hotspots      — WHICH fields get corrected most (where to invest)
  * confidence_histogram      — is the score distribution healthy or drifting?
  * status_breakdown          — operational view of the queue

`automatic_processing_rate` and `correction_rate` move in opposite directions and
together tell you whether a model change actually helped the business, not just
the offline metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.serving.db_models import InvoiceStatus
from src.serving.repository import InvoiceRepository

# Terminal statuses = an invoice the pipeline has actually decided on.
TERMINAL = {
    InvoiceStatus.AUTO_BOOKED.value,
    InvoiceStatus.NEEDS_REVIEW.value,
    InvoiceStatus.APPROVED.value,
    InvoiceStatus.REJECTED.value,
}

CONFIDENCE_BUCKETS = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]


@dataclass
class BusinessMetrics:
    total_processed: int = 0
    automatic_processing_rate: float = 0.0
    correction_rate: float = 0.0
    status_breakdown: dict[str, int] = field(default_factory=dict)
    field_error_hotspots: dict[str, int] = field(default_factory=dict)
    confidence_histogram: dict[str, int] = field(default_factory=dict)


class AnalyticsService:
    def __init__(self, repository: InvoiceRepository):
        self.repo = repository

    def compute(self) -> BusinessMetrics:
        by_status = self.repo.count_by_status()
        total = sum(c for s, c in by_status.items() if s in TERMINAL)

        auto = by_status.get(InvoiceStatus.AUTO_BOOKED.value, 0)
        corrected = self.repo.count_invoices_with_corrections()

        return BusinessMetrics(
            total_processed=total,
            automatic_processing_rate=round(auto / total, 3) if total else 0.0,
            correction_rate=round(corrected / total, 3) if total else 0.0,
            status_breakdown=by_status,
            field_error_hotspots=self.repo.correction_field_counts(),
            confidence_histogram=self._histogram(self.repo.confidences()),
        )

    @staticmethod
    def _histogram(values: list[float]) -> dict[str, int]:
        hist = {f"{lo:.2f}-{hi:.2f}": 0 for lo, hi in CONFIDENCE_BUCKETS}
        for v in values:
            for lo, hi in CONFIDENCE_BUCKETS:
                if lo <= v < hi:
                    hist[f"{lo:.2f}-{hi:.2f}"] += 1
                    break
        return hist
