"""Table / line-item detection stage (IPP-007).

Header fields (invoice number, total) appear once and suit text+regex. Line items
are different: they live in a 2-D table whose meaning comes from *geometry* —
which cell sits under which column header. Pure NLP loses that. So the industry
(and Zenvoices) frames line-item extraction as an object-detection problem: first
DETECT the table region, its rows, and its column bands as bounding boxes, THEN
read text inside each cell. That detection step is what a model like YOLO does.

We ship a deterministic, geometry-based simulated detector so the pipeline runs
with no model weights. The seam to the real thing is the `TableDetector`
Protocol: a `YoloTableDetector(TableDetector)` that returns row/column boxes from
an image would slot straight in — the assembly logic below is model-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.pipeline.ocr import OCRResult, OCRWord

# Column labels we expect in an invoice line-item table, with the header
# keywords that identify each column.
COLUMN_KEYWORDS = {
    "description": ("description", "item", "omschrijving"),
    "quantity": ("qty", "quantity", "aantal"),
    "unit_price": ("unit", "price", "prijs", "tarief"),
    "amount": ("amount", "total", "bedrag"),
}

RE_MONEY = re.compile(r"\d+[.,]\d{2}")
RE_QTY = re.compile(r"^\d+([.,]\d+)?$")


@dataclass
class LineItem:
    description: str | None = None
    quantity: str | None = None
    unit_price: str | None = None
    amount: str | None = None
    confidence: float = 0.0


@dataclass
class TableResult:
    line_items: list[LineItem] = field(default_factory=list)
    header_row_y: int | None = None
    column_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        return len(self.line_items)


class TableDetector:
    """Interface every table detector must satisfy."""

    def detect(self, ocr: OCRResult) -> TableResult: ...


class SimulatedTableDetector:
    """Geometry-based line-item detector over OCR words.

    Strategy (a stand-in for what a trained detector learns):
      1. Find the header row: the line containing the most column keywords.
      2. From the header words' x-positions, derive vertical column bands.
      3. Group the words below the header into rows by shared y (a line).
      4. Assign each word to a column by its x-centre, and only keep rows that
         carry a money `amount` (filters out notes/subtotal lines).
    """

    def detect(self, ocr: OCRResult) -> TableResult:
        header_y, columns = self._find_header(ocr)
        if header_y is None:
            return TableResult()  # no table found

        column_bounds = self._column_bounds(columns)
        column_centres = self._column_centres(columns)
        rows = self._group_rows_below(ocr, header_y)

        items: list[LineItem] = []
        for _, words in rows:
            if self._is_summary_row(words):
                continue  # subtotal / VAT / total belong to header fields, not items
            item = self._assign_row(words, column_centres)
            if item and item.amount:  # a real line item has a money amount
                items.append(item)

        return TableResult(
            line_items=items,
            header_row_y=header_y,
            column_bounds=column_bounds,
        )

    # --- steps ---------------------------------------------------------------

    SUMMARY_KEYWORDS = ("subtotal", "subtotaal", "vat", "btw", "total", "totaal", "currency")

    def _is_summary_row(self, words: list[OCRWord]) -> bool:
        text = " ".join(w.text.lower() for w in words)
        return any(kw in text for kw in self.SUMMARY_KEYWORDS)

    def _find_header(self, ocr: OCRResult) -> tuple[int | None, list[OCRWord]]:
        """The header is the row whose words hit the most column keywords."""
        best_y, best_hits, best_words = None, 0, []
        for y, words in ocr.rows():
            hits = sum(
                any(kw in w.text.lower() for kws in COLUMN_KEYWORDS.values() for kw in kws)
                for w in words
            )
            if hits > best_hits:
                best_y, best_hits, best_words = y, hits, words
        # Require at least two column headers to call it a table.
        return (best_y, best_words) if best_hits >= 2 else (None, [])

    def _column_bounds(self, header_words: list[OCRWord]) -> dict[str, tuple[int, int]]:
        """Map each detected header word to a column and its x-range."""
        bounds: dict[str, tuple[int, int]] = {}
        for w in sorted(header_words, key=lambda x: x.bbox[0]):
            col = self._match_column(w.text)
            if col and col not in bounds:
                # Column band spans from this header's left edge rightward; the
                # right edge is refined by the next column (or open-ended).
                bounds[col] = (w.bbox[0], w.bbox[2])
        # Extend each column's right edge to the next column's left edge.
        ordered = sorted(bounds.items(), key=lambda kv: kv[1][0])
        for i, (col, (x0, x1)) in enumerate(ordered):
            right = ordered[i + 1][1][0] if i + 1 < len(ordered) else 10_000
            bounds[col] = (x0, right)
        return bounds

    @staticmethod
    def _match_column(token: str) -> str | None:
        t = token.lower()
        for col, kws in COLUMN_KEYWORDS.items():
            if any(kw in t for kw in kws):
                return col
        return None

    def _column_centres(self, header_words: list[OCRWord]) -> dict[str, int]:
        """Each column's anchor point = the centre of its header word."""
        centres: dict[str, int] = {}
        for w in header_words:
            col = self._match_column(w.text)
            if col and col not in centres:
                centres[col] = (w.bbox[0] + w.bbox[2]) // 2
        return centres

    def _group_rows_below(self, ocr: OCRResult, header_y: int):
        return [(y, words) for y, words in ocr.rows() if y > header_y]

    def _assign_row(self, words: list[OCRWord], centres: dict[str, int]) -> LineItem | None:
        """Assign each word to the NEAREST column centre — robust to the small
        horizontal jitter that rigid column bands trip over."""
        if not centres:
            return None
        buckets: dict[str, list[str]] = {c: [] for c in centres}
        for w in words:
            x_centre = (w.bbox[0] + w.bbox[2]) // 2
            nearest = min(centres, key=lambda c: abs(centres[c] - x_centre))
            buckets[nearest].append(w.text)

        item = LineItem(
            description=" ".join(buckets.get("description", [])) or None,
            quantity=self._first(buckets.get("quantity", []), RE_QTY),
            unit_price=self._first(buckets.get("unit_price", []), RE_MONEY),
            amount=self._first(buckets.get("amount", []), RE_MONEY),
        )
        confs = [w.confidence for w in words]
        item.confidence = round(sum(confs) / len(confs), 3) if confs else 0.0
        return item

    @staticmethod
    def _first(tokens: list[str], pattern: re.Pattern) -> str | None:
        for t in tokens:
            if pattern.search(t):
                return t
        return None
