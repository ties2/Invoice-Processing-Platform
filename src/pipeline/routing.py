"""Routing decision engine (IPP-006).

This is where every upstream signal converges into a single, auditable decision:
should this invoice be booked automatically, or sent to a human?

Three independent signals feed the decision:
  1. overall confidence  (from OCR + extractor)
  2. validation result   (deterministic financial coherence)
  3. corner-case flag     (champion/challenger disagreement in shadow/canary)

Guiding rule: auto-booking requires ALL of {high confidence, no validation errors,
not a corner case}. Any single red flag routes to human review. Missing a fraud is
cheaper to prevent than to unwind, so the engine is deliberately conservative and,
crucially, records *why* it decided — the reasons are as important as the verdict.

The thresholds live in config, not code, because they are a business decision that
must be tuned against validation data and the cost of false positives vs. false
negatives — never a hardcoded constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline.extractor import FieldPrediction
from src.pipeline.validation import ValidationResult
from src.serving.config import settings
from src.serving.db_models import InvoiceStatus


@dataclass
class RoutingDecision:
    status: InvoiceStatus              # AUTO_BOOKED or NEEDS_REVIEW
    overall_confidence: float
    reasons: list[str] = field(default_factory=list)

    @property
    def is_auto(self) -> bool:
        return self.status == InvoiceStatus.AUTO_BOOKED


def aggregate_confidence(predictions: list[FieldPrediction]) -> float:
    """Overall confidence = the weakest field, not the average.

    A document is only as trustworthy as its least certain field: averaging would
    let a very confident supplier name mask a shaky total. Taking the minimum makes
    one bad field enough to trigger review — the safe choice for money.
    """
    if not predictions:
        return 0.0
    return round(min(p.confidence for p in predictions), 3)


class RoutingEngine:
    def __init__(
        self,
        auto_threshold: float | None = None,
        review_threshold: float | None = None,
    ):
        self.auto_threshold = auto_threshold or settings.AUTO_BOOK_THRESHOLD
        self.review_threshold = review_threshold or settings.REVIEW_THRESHOLD

    def decide(
        self,
        predictions: list[FieldPrediction],
        validation: ValidationResult,
        is_corner_case: bool = False,
    ) -> RoutingDecision:
        overall = aggregate_confidence(predictions)
        reasons: list[str] = []

        # Hard blocks first: these override confidence entirely.
        if validation.has_errors:
            reasons.append("validation_error")
        if is_corner_case:
            reasons.append("corner_case")

        # Confidence gate.
        if overall < self.auto_threshold:
            reasons.append(f"confidence {overall} < auto_threshold {self.auto_threshold}")

        # Auto-book only when there is nothing on the reasons list.
        if not reasons:
            return RoutingDecision(
                status=InvoiceStatus.AUTO_BOOKED,
                overall_confidence=overall,
                reasons=["all_checks_passed"],
            )

        return RoutingDecision(
            status=InvoiceStatus.NEEDS_REVIEW,
            overall_confidence=overall,
            reasons=reasons,
        )
