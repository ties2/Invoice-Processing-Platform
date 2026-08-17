"""LLM / VLM extractor (Path B).

Implements the same `Extractor` interface as `RuleBasedExtractor`, so it drops
into the pipeline, experiment, service and API with no other change — and can be
shadow-tested against the rule-based champion before it ever serves traffic.

Why an LLM/VLM here maps to where invoice products are heading in 2026: fields can
be described in natural language ("the customer's PO number, usually near the top
right"), which the rule-based extractor cannot do. The extractor is therefore
*prompt-configurable*.

Two safety ideas that matter for money:
  1. Grounding guard — an LLM can hallucinate a plausible total that is not on the
     page. Every returned value is checked against the OCR text; a value that
     does not appear in the source is heavily penalised in confidence. LLM
     self-reported confidence is poorly calibrated, so we never trust it alone.
  2. The neural output still passes through the deterministic validation gate
     downstream. The LLM proposes; arithmetic disposes.

The Anthropic client is injected (or lazily constructed), so tests run with a
fake client and no network. `LLMExtractor` uses OCR text; `VLMExtractor` uses the
page image via `OCRResult.source_image`.
"""

from __future__ import annotations

import base64
import json
import re

from src.pipeline.extractor import FieldPrediction
from src.pipeline.ocr import OCRResult
from src.serving.db_models import FieldName

# Default fields with natural-language hints — editable at construction time,
# which is the "describe the field you want" capability.
DEFAULT_FIELD_SPEC: dict[str, str] = {
    FieldName.INVOICE_NUMBER.value: "the invoice number / factuurnummer",
    FieldName.SUPPLIER.value: "the supplier / vendor company name",
    FieldName.INVOICE_DATE.value: "the invoice date",
    FieldName.VAT_RATE.value: "the VAT/BTW percentage, e.g. '21%'",
    FieldName.VAT_AMOUNT.value: "the VAT/BTW amount in currency",
    FieldName.SUBTOTAL.value: "the subtotal before VAT",
    FieldName.TOTAL.value: "the total amount due including VAT",
    FieldName.CURRENCY.value: "the ISO currency code, e.g. EUR or USD",
}

_SYSTEM = (
    "You extract structured fields from invoices. Return ONLY a JSON object, no "
    "prose, no markdown fences. For each requested field return an object with "
    '"value" (string, or null if absent) and "confidence" (0.0-1.0). Never invent '
    "values that are not present in the document."
)


class _BaseLLMExtractor:
    def __init__(
        self,
        field_spec: dict[str, str] | None = None,
        model: str = "claude-sonnet-4-6",
        client=None,
        grounding: bool = True,
    ):
        self.field_spec = field_spec or DEFAULT_FIELD_SPEC
        self.model = model
        self._client = client
        self.grounding = grounding

    # --- client (lazy, injectable) ------------------------------------------
    @property
    def client(self):
        if self._client is None:
            import anthropic  # lazy; only needed for the real path
            self._client = anthropic.Anthropic()
        return self._client

    # --- shared parsing / grounding -----------------------------------------
    def _field_instructions(self) -> str:
        return "\n".join(f"- {name}: {hint}" for name, hint in self.field_spec.items())

    @staticmethod
    def _parse_json(text: str) -> dict:
        # Be defensive: strip accidental fences / surrounding prose.
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object in model output: {text[:200]}")
        return json.loads(text[start : end + 1])

    def _to_predictions(self, parsed: dict, ocr_text: str) -> list[FieldPrediction]:
        preds: list[FieldPrediction] = []
        haystack = ocr_text.lower()
        for name in self.field_spec:
            entry = parsed.get(name) or {}
            value = entry.get("value")
            raw_conf = float(entry.get("confidence", 0.0) or 0.0)
            conf = max(0.0, min(1.0, raw_conf))

            if value and self.grounding:
                # Grounding guard: penalise values not found in the source text.
                grounded = str(value).lower() in haystack
                if not grounded:
                    conf = round(conf * 0.3, 3)  # likely hallucinated -> low trust
            preds.append(
                FieldPrediction(field_name=name, value=value, confidence=round(conf, 3), source="llm")
            )
        return preds


class LLMExtractor(_BaseLLMExtractor):
    """Text-based extraction: OCR text (with layout lines) -> structured JSON."""

    version = "llm-sonnet-v1"

    def extract(self, ocr: OCRResult) -> list[FieldPrediction]:
        document = "\n".join(ocr.lines)  # layout-preserving text
        prompt = (
            "Extract these fields from the invoice text below.\n\n"
            f"Fields:\n{self._field_instructions()}\n\n"
            f"Invoice text:\n{document}"
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        return self._to_predictions(self._parse_json(text), ocr.text)


class VLMExtractor(_BaseLLMExtractor):
    """Vision extraction: the page image itself -> structured JSON.

    Uses `OCRResult.source_image` (pixels), so it can read layout, stamps and
    tables that flat OCR text loses. Falls back with a clear error if no image
    was carried through.
    """

    version = "vlm-sonnet-v1"

    def extract(self, ocr: OCRResult) -> list[FieldPrediction]:
        if not ocr.source_image:
            raise ValueError("VLMExtractor requires OCRResult.source_image (page pixels)")

        b64 = base64.standard_b64encode(ocr.source_image).decode()
        prompt = (
            "Extract these fields from the invoice image.\n\n"
            f"Fields:\n{self._field_instructions()}"
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        text = resp.content[0].text
        # Grounding uses OCR text if present; otherwise trust is model-only.
        return self._to_predictions(self._parse_json(text), ocr.text)
