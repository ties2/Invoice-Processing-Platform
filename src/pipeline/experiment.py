"""Extraction experiment layer (IPP-004).

This is the direct answer to three production pains at once:

  1. Legacy modernisation  — the legacy extractor is the `champion`; a new one is
     the `challenger`, both behind the same `Extractor` interface (Strangler Fig).
  2. Safe A/B testing on real traffic — `SHADOW` mode runs the challenger on the
     *same* live invoices as the champion but its output never affects the
     decision, so a new system can be validated on millions of real invoices at
     zero production risk. `CANARY` mode then serves a small, deterministic slice
     of traffic from the challenger.
  3. Corner-case discovery — every place the champion and challenger *disagree*
     is a candidate corner case, automatically surfaced for human labelling.

The champion→challenger→promote flow mirrors the reference project's
staging→production lifecycle: when the shadow metrics say the challenger wins,
you promote it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from src.pipeline.extractor import Extractor, FieldPrediction
from src.pipeline.ocr import OCRResult


class ExperimentMode(str, Enum):
    CHAMPION_ONLY = "champion_only"  # no challenger involved
    SHADOW = "shadow"                # challenger runs but never serves
    CANARY = "canary"                # a % of traffic is served by the challenger


@dataclass
class FieldComparison:
    field_name: str
    champion_value: str | None
    challenger_value: str | None
    champion_conf: float
    challenger_conf: float
    agree: bool


@dataclass
class ExperimentReport:
    mode: ExperimentMode
    champion_version: str
    challenger_version: str | None
    served_by: str  # "champion" | "challenger"
    comparisons: list[FieldComparison] = field(default_factory=list)
    agreement_rate: float = 1.0
    is_corner_case: bool = False

    @property
    def disagreements(self) -> list[FieldComparison]:
        """The fields where champion and challenger differ — candidate corner cases."""
        return [c for c in self.comparisons if not c.agree]


def _normalise(value: str | None) -> str | None:
    return value.strip().lower() if isinstance(value, str) else value


class ExtractionExperiment:
    """Runs champion (and optionally challenger) and reports on the comparison."""

    def __init__(
        self,
        champion: Extractor,
        challenger: Extractor | None = None,
        mode: ExperimentMode = ExperimentMode.CHAMPION_ONLY,
        canary_pct: int = 0,
        corner_case_conf: float = 0.80,
    ):
        self.champion = champion
        self.challenger = challenger
        self.mode = mode
        self.canary_pct = max(0, min(100, canary_pct))
        self.corner_case_conf = corner_case_conf

        if mode in (ExperimentMode.SHADOW, ExperimentMode.CANARY) and challenger is None:
            raise ValueError(f"{mode.value} mode requires a challenger extractor")

    def _canary_selects_challenger(self, routing_key: str) -> bool:
        """Deterministic bucketing: the same invoice always lands in the same arm,
        so a given supplier/document sees a consistent result (no flip-flopping)."""
        bucket = int(hashlib.md5(routing_key.encode()).hexdigest(), 16) % 100
        return bucket < self.canary_pct

    def run(
        self, ocr: OCRResult, routing_key: str | None = None
    ) -> tuple[list[FieldPrediction], ExperimentReport]:
        champion_preds = self.champion.extract(ocr)

        # Champion-only: nothing to compare, champion always serves.
        if self.mode == ExperimentMode.CHAMPION_ONLY or self.challenger is None:
            report = ExperimentReport(
                mode=self.mode,
                champion_version=self.champion.version,
                challenger_version=None,
                served_by="champion",
                is_corner_case=self._low_confidence(champion_preds),
            )
            return champion_preds, report

        # Shadow / canary: run the challenger too.
        challenger_preds = self.challenger.extract(ocr)
        comparisons, agreement_rate = self._compare(champion_preds, challenger_preds)

        # Decide who actually serves the result.
        if self.mode == ExperimentMode.CANARY and routing_key is not None \
                and self._canary_selects_challenger(routing_key):
            served_by, serving_preds = "challenger", challenger_preds
        else:
            # Shadow always serves champion; canary serves champion for the rest.
            served_by, serving_preds = "champion", champion_preds

        report = ExperimentReport(
            mode=self.mode,
            champion_version=self.champion.version,
            challenger_version=self.challenger.version,
            served_by=served_by,
            comparisons=comparisons,
            agreement_rate=agreement_rate,
            is_corner_case=(
                len([c for c in comparisons if not c.agree]) > 0
                or self._low_confidence(serving_preds)
            ),
        )
        return serving_preds, report

    def _compare(
        self, champ: list[FieldPrediction], chall: list[FieldPrediction]
    ) -> tuple[list[FieldComparison], float]:
        chall_by_name = {p.field_name: p for p in chall}
        comparisons: list[FieldComparison] = []
        for cp in champ:
            xp = chall_by_name.get(cp.field_name)
            agree = xp is not None and _normalise(cp.value) == _normalise(xp.value)
            comparisons.append(
                FieldComparison(
                    field_name=cp.field_name,
                    champion_value=cp.value,
                    challenger_value=xp.value if xp else None,
                    champion_conf=cp.confidence,
                    challenger_conf=xp.confidence if xp else 0.0,
                    agree=agree,
                )
            )
        agreement_rate = (
            sum(c.agree for c in comparisons) / len(comparisons) if comparisons else 1.0
        )
        return comparisons, round(agreement_rate, 3)

    def _low_confidence(self, preds: list[FieldPrediction]) -> bool:
        return any(p.confidence < self.corner_case_conf for p in preds)
