"""LLM extractor tests using a fake Anthropic client (no network)."""
import json
from types import SimpleNamespace

from src.pipeline.ocr import SimulatedOCR
from src.pipeline.extractor_llm import LLMExtractor


class FakeClient:
    """Mimics client.messages.create(...).content[0].text with canned JSON."""
    def __init__(self, payload: dict):
        self._payload = payload
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        text = json.dumps(self._payload)
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


def _ocr(raw):
    return SimulatedOCR(noise=0.0).recognize(raw)


def test_llm_extractor_parses_structured_output(clean_invoice_bytes):
    payload = {
        "invoice_number": {"value": "INV-2026-1234", "confidence": 0.98},
        "total": {"value": "1210,00", "confidence": 0.97},
        "supplier": {"value": "KPN B.V.", "confidence": 0.95},
    }
    ex = LLMExtractor(client=FakeClient(payload))
    out = {p.field_name: p for p in ex.extract(_ocr(clean_invoice_bytes))}
    assert out["invoice_number"].value == "INV-2026-1234"
    assert out["invoice_number"].source == "llm"
    assert out["total"].confidence > 0.9  # grounded (present in OCR text)


def test_grounding_guard_penalises_hallucination(clean_invoice_bytes):
    # The model claims a total that is NOT on the page -> confidence must drop.
    payload = {"total": {"value": "9999,99", "confidence": 0.99}}
    ex = LLMExtractor(client=FakeClient(payload))
    out = {p.field_name: p for p in ex.extract(_ocr(clean_invoice_bytes))}
    assert out["total"].value == "9999,99"
    assert out["total"].confidence < 0.4  # heavily penalised


def test_handles_fenced_json(clean_invoice_bytes):
    class FencedClient(FakeClient):
        def _create(self, **kwargs):
            text = "```json\n" + json.dumps({"currency": {"value": "EUR", "confidence": 0.9}}) + "\n```"
            return SimpleNamespace(content=[SimpleNamespace(text=text)])
    ex = LLMExtractor(client=FencedClient({}))
    out = {p.field_name: p for p in ex.extract(_ocr(clean_invoice_bytes))}
    assert out["currency"].value == "EUR"
