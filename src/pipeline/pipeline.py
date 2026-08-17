"""Extraction pipeline orchestrator (IPP-009).

Composes every stage into a single, testable `run()` call so the service layer
stays thin and the ordering of stages lives in exactly one place:

    OCR -> (experiment: champion/challenger extract) -> table detect
        -> validate -> route -> outcome

The pipeline always runs through an `ExtractionExperiment`. In the ordinary case
that experiment is CHAMPION_ONLY (just the current extractor); flip it to SHADOW
or CANARY to test a challenger with no change to the service or API. That single
choice is the whole legacy-modernisation seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline.experiment import ExperimentMode, ExperimentReport, ExtractionExperiment
from src.pipeline.extractor import Extractor, FieldPrediction, RuleBasedExtractor
from src.pipeline.ocr import OCREngine, SimulatedOCR
from src.pipeline.routing import RoutingDecision, RoutingEngine
from src.pipeline.table_detector import LineItem, SimulatedTableDetector, TableDetector
from src.pipeline.validation import InvoiceValidator, ValidationResult
from src.serving.db_models import FieldName


@dataclass
class PipelineOutcome:
    predictions: list[FieldPrediction]
    line_items: list[LineItem]
    validation: ValidationResult
    decision: RoutingDecision
    extractor_version: str
    served_by: str
    report: ExperimentReport
    ocr_mean_confidence: float = 0.0

    def field_map(self) -> dict[str, str | None]:
        return {p.field_name: p.value for p in self.predictions}

    @property
    def supplier(self) -> str | None:
        return self.field_map().get(FieldName.SUPPLIER.value)


class ExtractionPipeline:
    def __init__(
        self,
        ocr: OCREngine | None = None,
        experiment: ExtractionExperiment | None = None,
        table_detector: TableDetector | None = None,
        validator: InvoiceValidator | None = None,
        routing_engine: RoutingEngine | None = None,
    ):
        self.ocr = ocr or SimulatedOCR()
        self.experiment = experiment or ExtractionExperiment(
            champion=RuleBasedExtractor(), mode=ExperimentMode.CHAMPION_ONLY
        )
        self.table_detector = table_detector or SimulatedTableDetector()
        self.validator = validator or InvoiceValidator()
        self.routing_engine = routing_engine or RoutingEngine()

    @classmethod
    def with_extractor(cls, extractor: Extractor, **kw) -> "ExtractionPipeline":
        """Convenience: build a CHAMPION_ONLY pipeline around one extractor."""
        return cls(
            experiment=ExtractionExperiment(champion=extractor, mode=ExperimentMode.CHAMPION_ONLY),
            **kw,
        )

    def run(self, document_bytes: bytes, routing_key: str | None = None) -> PipelineOutcome:
        ocr_result = self.ocr.recognize(document_bytes)

        predictions, report = self.experiment.run(ocr_result, routing_key=routing_key)
        table = self.table_detector.detect(ocr_result)

        validation = self.validator.validate({p.field_name: p.value for p in predictions})
        decision = self.routing_engine.decide(
            predictions, validation, is_corner_case=report.is_corner_case
        )

        return PipelineOutcome(
            predictions=predictions,
            line_items=table.line_items,
            validation=validation,
            decision=decision,
            extractor_version=report.served_by == "challenger"
            and (report.challenger_version or "")
            or report.champion_version,
            served_by=report.served_by,
            report=report,
            ocr_mean_confidence=ocr_result.mean_confidence,
        )
