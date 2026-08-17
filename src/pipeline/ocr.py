"""OCR stage (IPP-003).

The OCR engine turns raw document bytes into structured text: a list of words,
each with a bounding box and a per-word confidence. That output contract is
exactly what real engines (Tesseract's `image_to_data`, PaddleOCR) return, and
exactly what layout-aware models (LayoutLMv3, Donut) consume downstream.

We ship a deterministic *simulated* engine so the whole pipeline runs anywhere
with no native OCR dependency. The seam to a real engine is the `OCREngine`
Protocol — a `TesseractOCR(OCREngine)` class would slot in with no change to the
extractor, service, or API.

Simulation contract: an "invoice artifact" is a UTF-8 text blob representing what
OCR would have read off the page. The simulated engine tokenises it, assigns
synthetic bounding boxes (left-to-right, top-to-bottom), and a per-word
confidence that can be degraded to mimic a noisy scan.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


@dataclass
class OCRWord:
    """A single recognised token with its geometry and confidence."""

    text: str
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixels
    confidence: float                # 0.0 - 1.0


@dataclass
class OCRResult:
    """Full OCR output for one document."""

    words: list[OCRWord] = field(default_factory=list)
    # Optional raw page image (PNG/JPEG bytes). OCR engines set this when a
    # rendered page is available, so a downstream vision model (VLM) can consume
    # pixels through the same OCRResult the text pipeline already uses.
    source_image: bytes | None = None

    @property
    def text(self) -> str:
        """Reconstructed plain text (what a text-only extractor would use)."""
        return " ".join(w.text for w in self.words)

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    def _median_height(self) -> int:
        if not self.words:
            return 20
        heights = sorted(w.bbox[3] - w.bbox[1] for w in self.words)
        return max(1, heights[len(heights) // 2])

    def rows(self, tolerance: int | None = None) -> list[tuple[int, list[OCRWord]]]:
        """Cluster words into visual rows by y-proximity, not exact equality.

        Real OCR gives words on the same line slightly different `top` values, so
        grouping by exact y shatters lines. We cluster any words whose tops fall
        within a tolerance (default: half the median glyph height). Each row is
        returned as (row_top, words-left-to-right), ordered top-to-bottom.
        """
        if not self.words:
            return []
        tol = tolerance if tolerance is not None else max(4, self._median_height() // 2)

        by_top = sorted(self.words, key=lambda w: w.bbox[1])
        rows: list[tuple[int, list[OCRWord]]] = []
        current_top = by_top[0].bbox[1]
        bucket: list[OCRWord] = []
        for w in by_top:
            if w.bbox[1] - current_top <= tol:
                bucket.append(w)
            else:
                rows.append((current_top, sorted(bucket, key=lambda x: x.bbox[0])))
                bucket = [w]
                current_top = w.bbox[1]
        rows.append((current_top, sorted(bucket, key=lambda x: x.bbox[0])))
        return rows

    @property
    def lines(self) -> list[str]:
        """Text lines reconstructed from geometry (jitter-tolerant)."""
        return [" ".join(w.text for w in words) for _, words in self.rows()]


class OCREngine:
    """Interface every OCR backend must satisfy."""

    def recognize(self, document_bytes: bytes) -> OCRResult: ...


class SimulatedOCR:
    """Deterministic OCR simulation over a UTF-8 text artifact.

    Deterministic because we derive per-word confidence from a hash of the token
    and line, so the same document always yields the same result (reproducible
    tests, no random flakiness). A `noise` factor lets us demonstrate the
    low-confidence -> human-review path on demand.
    """

    def __init__(self, noise: float = 0.0, base_confidence: float = 0.99):
        # `noise` in [0, 1] pushes confidences down to exercise the review path.
        self.noise = max(0.0, min(1.0, noise))
        self.base_confidence = base_confidence

    def _word_confidence(self, token: str, line_idx: int, col_idx: int) -> float:
        # Stable pseudo-random jitter derived from the token itself.
        seed = f"{token}|{line_idx}|{col_idx}".encode()
        jitter = int(hashlib.md5(seed).hexdigest(), 16) % 1000 / 1000.0  # 0..0.999
        # A clean digital PDF reads at near-1.0; scans (noise>0) degrade sharply.
        conf = self.base_confidence - 0.015 * jitter
        conf -= self.noise * (0.4 + 0.4 * jitter)
        return round(max(0.05, min(0.999, conf)), 3)

    def recognize(self, document_bytes: bytes) -> OCRResult:
        try:
            raw_text = document_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            # A real engine would still try; here a non-text artifact is a hard
            # failure the pipeline can map to a FAILED invoice status.
            raise ValueError("SimulatedOCR expects a UTF-8 text artifact") from exc

        words: list[OCRWord] = []
        line_height = 20
        char_width = 9

        for line_idx, line in enumerate(raw_text.splitlines()):
            y0 = line_idx * line_height
            # Derive each word's x from its actual character offset in the line,
            # so space-aligned columns keep their real horizontal geometry (this
            # is what lets the table detector recover columns downstream).
            for col_idx, match in enumerate(re.finditer(r"\S+", line)):
                token = match.group()
                x0 = match.start() * char_width
                x1 = match.end() * char_width
                words.append(
                    OCRWord(
                        text=token,
                        bbox=(x0, y0, x1, y0 + line_height),
                        confidence=self._word_confidence(token, line_idx, col_idx),
                    )
                )

        return OCRResult(words=words)
