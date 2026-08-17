"""Observability: structured logging (IPP-008).

JSON logs so every processing decision is queryable in production (filter by
status, supplier, extractor_version, correlation id). In a system that books
millions of invoices, "why did invoice 12345 auto-book?" must be answerable from
logs alone — which is why routing decisions carry their `reasons`.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any structured extras passed via logger.info(..., extra={"extra": {...}}).
        if hasattr(record, "extra"):
            payload.update(record.extra)  # type: ignore[attr-defined]
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


logger = logging.getLogger("ipp")


def log_processing_decision(invoice_id: int, outcome) -> None:
    """Emit one structured line capturing the whole decision, incl. the reasons."""
    logger.info(
        "invoice_processed",
        extra={"extra": {
            "invoice_id": invoice_id,
            "status": outcome.decision.status.value,
            "overall_confidence": outcome.decision.overall_confidence,
            "reasons": outcome.decision.reasons,
            "extractor_version": outcome.extractor_version,
            "served_by": outcome.served_by,
            "supplier": outcome.supplier,
            "validation_errors": [i.rule for i in outcome.validation.issues if i.severity.value == "error"],
            "n_line_items": len(outcome.line_items),
        }},
    )
