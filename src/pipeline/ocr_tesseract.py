"""Real OCR backend using Tesseract (Path B).

Implements the same `OCREngine` interface as `SimulatedOCR`, so the service,
extractor, routing and API are completely unaware of the swap — this is the whole
payoff of programming to the interface.

Pipeline: document bytes (image or PDF) -> PIL image(s) -> `pytesseract
image_to_data` -> words with bounding boxes and per-word confidence, i.e. exactly
the `OCRResult` contract the rest of the system already consumes.

Heavy/native dependencies (pytesseract, Pillow, pdf2image + the tesseract and
poppler binaries) are imported lazily so the base project still runs and tests
without them installed.
"""

from __future__ import annotations

import io

from src.pipeline.ocr import OCRResult, OCRWord

# Tesseract reports confidence as 0-100 (or -1 for non-text); we normalise to 0-1.
_MIN_CONF = 0.0


class TesseractOCR:
    """OCR via the Tesseract engine. Accepts PNG/JPG bytes or PDF bytes."""

    def __init__(self, lang: str = "eng+nld", dpi: int = 300, min_confidence: float = 0.0):
        # eng+nld: invoices here are English/Dutch; Tesseract can load both.
        self.lang = lang
        self.dpi = dpi
        self.min_confidence = min_confidence

    def _load_images(self, document_bytes: bytes):
        """Return a list of PIL images (one per page for PDFs, else one)."""
        from PIL import Image  # lazy

        # PDFs start with the '%PDF' magic bytes.
        if document_bytes[:4] == b"%PDF":
            from pdf2image import convert_from_bytes  # lazy
            return convert_from_bytes(document_bytes, dpi=self.dpi)
        return [Image.open(io.BytesIO(document_bytes))]

    def recognize(self, document_bytes: bytes) -> OCRResult:
        import pytesseract  # lazy
        from pytesseract import Output

        words: list[OCRWord] = []
        page_y_offset = 0

        for image in self._load_images(document_bytes):
            data = pytesseract.image_to_data(
                image, lang=self.lang, output_type=Output.DICT
            )
            n = len(data["text"])
            for i in range(n):
                text = data["text"][i].strip()
                raw_conf = float(data["conf"][i])
                if not text or raw_conf < 0:
                    continue  # skip layout boxes / empty tokens
                conf = max(_MIN_CONF, min(1.0, raw_conf / 100.0))
                if conf < self.min_confidence:
                    continue
                x, y = data["left"][i], data["top"][i] + page_y_offset
                w, h = data["width"][i], data["height"][i]
                words.append(
                    OCRWord(
                        text=text,
                        bbox=(x, y, x + w, y + h),
                        confidence=round(conf, 3),
                    )
                )
            # Stack multi-page PDFs vertically so `lines` stays ordered per page.
            page_y_offset += image.height + 50

        return OCRResult(words=words)
