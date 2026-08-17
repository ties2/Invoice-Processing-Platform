"""Field extraction stage (IPP-003).

Turns an `OCRResult` into a list of `FieldPrediction`s — one per invoice field,
each carrying its own confidence. Confidence is the product of two signals:

  1. match quality  — did we hit a strong anchor keyword + a well-formed value,
     or only a loose fallback pattern?
  2. OCR confidence — how sure was OCR about the exact tokens we matched?

That per-field confidence is what the routing stage later uses to decide
auto-book vs. human review.

Baseline = `RuleBasedExtractor` (regex/anchors). It is deliberately the "small
component": deterministic, cheap, debuggable, and a strong baseline to beat. The
`Extractor` Protocol is the seam where a `LayoutLMExtractor` or an
`LLMExtractor` would slot in — the service, API and DB never change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.pipeline.ocr import OCRResult
from src.serving.db_models import FieldName

EXTRACTOR_VERSION = "rule-based-v1"


@dataclass
class FieldPrediction:
    field_name: str
    value: str | None
    confidence: float
    source: str = "regex"


class Extractor:
    """Interface every extractor must satisfy."""

    version: str

    def extract(self, ocr: OCRResult) -> list[FieldPrediction]: ...


class RuleBasedExtractor:
    """Anchor-keyword + regex baseline extractor."""

    version = EXTRACTOR_VERSION

    # Legal-entity suffixes that make a line much more likely to be a supplier.
    ENTITY_SUFFIX = re.compile(r"\b(b\.?v\.?|n\.?v\.?|ltd|gmbh|inc|llc|s\.?a\.?)\b", re.I)

    def __init__(self, known_suppliers: set[str] | None = None):
        # A supplier master list. A registry *hit* is a reliable signal (high
        # confidence); a mere heuristic guess is not. Unknown suppliers therefore
        # route to review — and once corrected, they can be added here: the
        # "new supplier -> review -> learn" loop in miniature.
        self.known_suppliers = {s.lower() for s in (known_suppliers or set())}

    # Anchor keywords per field. A hit near one of these raises confidence.
    ANCHORS = {
        FieldName.INVOICE_NUMBER.value: r"(invoice|factuur)\s*(no\.?|number|nummer|#)",
        FieldName.INVOICE_DATE.value: r"(date|datum)",
        FieldName.VAT_RATE.value: r"(vat|btw)",
        FieldName.VAT_AMOUNT.value: r"(vat|btw)\s*(amount|bedrag)?",
        FieldName.SUBTOTAL.value: r"\b(subtotal|subtotaal)\b",
        FieldName.TOTAL.value: r"\b(total|totaal|amount due|te betalen)\b",
    }

    # Value patterns.
    RE_INVOICE_NO = re.compile(r"\b([A-Z]{2,}[-/]?\d{3,}[-/]?\d*)\b")
    RE_DATE = re.compile(
        r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b"
    )
    RE_PERCENT = re.compile(r"\b(\d{1,2})\s*%")
    # Money must end in a 2-digit decimal, so we never start matching in the
    # middle of a number (e.g. "1000,00" -> "1000,00", never "000,00").
    #   alt 1: grouped thousands + decimal  -> 1.250,00 / 1,250.00
    #   alt 2: plain integer + decimal      -> 1000,00 / 210,00
    RE_MONEY = re.compile(
        r"(\d{1,3}(?:[.,]\d{3})+[.,]\d{2}|\d+[.,]\d{2})"
    )
    RE_CURRENCY = re.compile(r"(€|\beur\b|\$|\busd\b)", re.IGNORECASE)

    def extract(self, ocr: OCRResult) -> list[FieldPrediction]:
        text = ocr.text
        low = text.lower()
        ocr_conf = ocr.mean_confidence

        preds: list[FieldPrediction] = []

        preds.append(self._invoice_number(text, low, ocr_conf))
        preds.append(self._supplier(ocr, ocr_conf))
        preds.append(self._date(text, low, ocr_conf))
        preds.append(self._vat_rate(text, low, ocr_conf))
        preds.extend(self._amounts(ocr.lines, ocr_conf))
        preds.append(self._currency(text, ocr_conf))

        return preds

    # --- helpers -------------------------------------------------------------

    def _anchored(self, low: str, field: str) -> bool:
        pat = self.ANCHORS.get(field)
        return bool(pat and re.search(pat, low))

    def _score(self, matched: bool, anchored: bool, ocr_conf: float) -> float:
        """Combine match quality with OCR confidence into a field confidence."""
        if not matched:
            return 0.0
        base = 0.98 if anchored else 0.72  # anchored hits are far more trustworthy
        return round(base * ocr_conf, 3)

    def _invoice_number(self, text, low, ocr_conf) -> FieldPrediction:
        m = self.RE_INVOICE_NO.search(text)
        val = m.group(1) if m else None
        conf = self._score(bool(val), self._anchored(low, FieldName.INVOICE_NUMBER.value), ocr_conf)
        return FieldPrediction(FieldName.INVOICE_NUMBER.value, val, conf)

    def _supplier(self, ocr: OCRResult, ocr_conf) -> FieldPrediction:
        # Baseline heuristic: the supplier name is usually the first line.
        rows = ocr.rows()
        first_line_words = [w.text for w in rows[0][1]] if rows else []
        val = " ".join(first_line_words) if first_line_words else None
        if not val:
            return FieldPrediction(FieldName.SUPPLIER.value, None, 0.0, source="heuristic")

        norm = val.lower().strip()
        if norm in self.known_suppliers:
            # Registry hit: reliable.
            conf, source = round(0.99 * ocr_conf, 3), "registry"
        elif self.ENTITY_SUFFIX.search(val):
            # Looks like a company name but unverified.
            conf, source = round(0.80 * ocr_conf, 3), "heuristic"
        else:
            # A weak guess; say so honestly with low confidence.
            conf, source = round(0.55 * ocr_conf, 3), "heuristic"
        return FieldPrediction(FieldName.SUPPLIER.value, val, conf, source=source)

    def _date(self, text, low, ocr_conf) -> FieldPrediction:
        m = self.RE_DATE.search(text)
        val = m.group(1) if m else None
        conf = self._score(bool(val), self._anchored(low, FieldName.INVOICE_DATE.value), ocr_conf)
        return FieldPrediction(FieldName.INVOICE_DATE.value, val, conf)

    def _vat_rate(self, text, low, ocr_conf) -> FieldPrediction:
        m = self.RE_PERCENT.search(text)
        val = f"{m.group(1)}%" if m else None
        conf = self._score(bool(val), self._anchored(low, FieldName.VAT_RATE.value), ocr_conf)
        return FieldPrediction(FieldName.VAT_RATE.value, val, conf)

    def _amounts(self, lines: list[str], ocr_conf) -> list[FieldPrediction]:
        """Find subtotal / VAT amount / total by anchoring on their line labels."""
        results: list[FieldPrediction] = []
        for field in (FieldName.SUBTOTAL.value, FieldName.VAT_AMOUNT.value, FieldName.TOTAL.value):
            value, anchored = self._amount_near_anchor(lines, field)
            conf = self._score(value is not None, anchored, ocr_conf)
            results.append(FieldPrediction(field, value, conf))
        return results

    def _amount_near_anchor(self, lines: list[str], field: str) -> tuple[str | None, bool]:
        """For each reconstructed line: if it carries the field's label, take the
        money value on that same line (the last one, to skip a VAT *rate* like
        21% and land on the VAT *amount*). Falls back to None."""
        anchor = self.ANCHORS.get(field)
        for line in lines:
            if anchor and re.search(anchor, line.lower()):
                matches = self.RE_MONEY.findall(line)
                if matches:
                    return self._normalize_money(matches[-1]), True
        return None, False

    @staticmethod
    def _normalize_money(raw: str) -> str:
        """Normalise '1.250,00' or '1,250.00' to '1250.00'."""
        s = raw.strip()
        if "," in s and "." in s:
            # The last separator is the decimal one.
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            # Comma as decimal (European) if it has exactly 2 trailing digits.
            s = s.replace(",", ".") if re.search(r",\d{2}$", s) else s.replace(",", "")
        return s

    def _currency(self, text, ocr_conf) -> FieldPrediction:
        m = self.RE_CURRENCY.search(text)
        raw = m.group(1).lower() if m else None
        mapping = {"€": "EUR", "eur": "EUR", "$": "USD", "usd": "USD"}
        val = mapping.get(raw) if raw else None
        conf = round(0.97 * ocr_conf, 3) if val else 0.0
        return FieldPrediction(FieldName.CURRENCY.value, val, conf)


# ---------------------------------------------------------------------------
# LegacyExtractor (IPP-004): a simulated "2016" extractor we are modernising.
# ---------------------------------------------------------------------------
class LegacyExtractor:
    """A deliberately brittle extractor, standing in for the legacy 2016 code.

    It reproduces the exact class of bug we fixed in the modern extractor: it
    reads the *flattened* OCR text (ignoring line/layout structure) and grabs the
    first money value it sees for every amount field. On a real multi-line
    invoice this silently mislabels subtotal/VAT/total — a textbook production
    corner case. Kept behind the same `Extractor` interface so it can be wrapped,
    shadow-tested against a challenger, and eventually strangled out.
    """

    version = "legacy-2016"

    RE_INVOICE_NO = re.compile(r"\b([A-Z]{2,}\d+)\b")  # naive: stops at first digit run
    RE_DATE = re.compile(r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b")
    RE_MONEY = re.compile(r"(\d+[.,]\d{2})")

    def extract(self, ocr: OCRResult) -> list[FieldPrediction]:
        text = ocr.text  # NOTE: flattened — the legacy layout bug lives here.
        conf = 0.90  # legacy code reported a flat, over-confident score

        inv = self.RE_INVOICE_NO.search(text)
        date = self.RE_DATE.search(text)
        first_money = self.RE_MONEY.search(text)
        money_val = self._legacy_money(first_money.group(1)) if first_money else None

        return [
            FieldPrediction(FieldName.INVOICE_NUMBER.value, inv.group(1) if inv else None, conf, "legacy"),
            # legacy never extracted these:
            FieldPrediction(FieldName.SUPPLIER.value, None, 0.0, "legacy"),
            FieldPrediction(FieldName.INVOICE_DATE.value, date.group(1) if date else None, conf, "legacy"),
            FieldPrediction(FieldName.VAT_RATE.value, None, 0.0, "legacy"),
            # the bug: the SAME first money value is used for every amount field
            FieldPrediction(FieldName.SUBTOTAL.value, money_val, conf, "legacy"),
            FieldPrediction(FieldName.VAT_AMOUNT.value, money_val, conf, "legacy"),
            FieldPrediction(FieldName.TOTAL.value, money_val, conf, "legacy"),
            FieldPrediction(FieldName.CURRENCY.value, None, 0.0, "legacy"),
        ]

    @staticmethod
    def _legacy_money(raw: str) -> str:
        # Legacy normalisation was naive: just swap comma for dot.
        return raw.replace(".", "").replace(",", ".") if "," in raw else raw
