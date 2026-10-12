"""Tests for the RecordedLLMClient replay path and LLM-in-shadow experiment."""
import json

from src.pipeline.ocr import SimulatedOCR
from src.pipeline.extractor import RuleBasedExtractor
from src.pipeline.extractor_llm import (
    LLMExtractor, RecordedLLMClient, document_key,
)
from src.pipeline.experiment import ExtractionExperiment, ExperimentMode


def _ocr(raw: str):
    return SimulatedOCR(noise=0.0).recognize(raw)


def _cache_for(ocr, payload: dict) -> RecordedLLMClient:
    """Build a one-entry replay cache keyed exactly as the extractor will key it."""
    document = "\n".join(ocr.lines)
    return RecordedLLMClient({document_key(document): json.dumps(payload)})


def test_recorded_client_replays_by_document_key(clean_invoice_bytes):
    ocr = _ocr(clean_invoice_bytes)
    client = _cache_for(ocr, {"supplier": {"value": "KPN B.V.", "confidence": 0.95}})
    out = {p.field_name: p for p in LLMExtractor(client=client).extract(ocr)}
    assert out["supplier"].value == "KPN B.V."
    assert out["supplier"].source == "llm"


def test_recorded_client_raises_on_cache_miss(clean_invoice_bytes):
    ocr = _ocr(clean_invoice_bytes)
    empty = RecordedLLMClient({})
    try:
        LLMExtractor(client=empty).extract(ocr)
    except KeyError:
        return
    raise AssertionError("expected a cache-miss KeyError")


def test_llm_challenger_runs_in_shadow_without_serving(clean_invoice_bytes):
    ocr = _ocr(clean_invoice_bytes)
    champion = RuleBasedExtractor()
    challenger = LLMExtractor(
        client=_cache_for(ocr, {"supplier": {"value": "ACME GMBH", "confidence": 0.9}})
    )
    exp = ExtractionExperiment(
        champion=champion, challenger=challenger, mode=ExperimentMode.SHADOW,
    )
    served, report = exp.run(ocr, routing_key="doc-1")

    # Shadow mode: the champion's output is always what serves.
    assert report.served_by == "champion"
    champ_supplier = {p.field_name: p.value for p in champion.extract(ocr)}["supplier"]
    served_supplier = {p.field_name: p.value for p in served}["supplier"]
    assert served_supplier == champ_supplier
    # ...but the challenger was still run and compared.
    assert report.challenger_version == challenger.version
    assert report.comparisons
