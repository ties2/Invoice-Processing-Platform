"""Real-invoice dataset loader: ICDAR-2019 SROIE (IPP-010).

Replaces the toy `sample_data/*.txt` blobs with real scanned receipts so the
Tesseract OCR path, the extractor and the router run end-to-end on genuine
pages — and so extraction can be *scored* against ground-truth labels.

Dataset layout (both the ICDAR SROIE release and the common Kaggle mirrors use
this shape):

    <root>/
      img/   0001.jpg ...        scanned receipt images
      box/   0001.csv ...        per-word boxes: x1,y1,x2,y2,x3,y3,x4,y4,text
      key/   0001.json ...       entities: {company, date, address, total}

Where to get it (pick one, then point IPP_DATASET_ROOT at the folder):
  • Kaggle:  `kaggle datasets download -d urbikn/sroie-datasetv2` then unzip
  • GitHub:  github.com/zzzDavid/ICDAR-2019-SROIE  (data/img, data/box, data/key)
  • Official: rrc.cvc.uab.es/?ch=13  (ICDAR 2019 Robust Reading, Task 1-3)

The SROIE entity names are mapped onto this project's field names. SROIE labels
only company/date/total, so unlabeled fields (invoice_number, VAT, subtotal)
are simply not scored — never counted as wrong.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# SROIE entity  ->  this project's FieldName value
SROIE_TO_FIELD = {
    "company": "supplier",
    "date": "invoice_date",
    "total": "total",
    # "address" has no field in this pipeline; kept in ground-truth for display.
}


@dataclass
class GroundTruthBox:
    text: str
    bbox: tuple[int, int, int, int]  # axis-aligned (x0, y0, x1, y1)


@dataclass
class InvoiceSample:
    sample_id: str
    image_path: str
    ground_truth: dict[str, str]              # field_name -> expected value
    word_boxes: list[GroundTruthBox] = field(default_factory=list)

    @property
    def image_bytes(self) -> bytes:
        with open(self.image_path, "rb") as fh:
            return fh.read()


class SroieDataset:
    """Iterate real receipts as (image_bytes, ground_truth) samples."""

    def __init__(self, root: str | os.PathLike | None = None):
        root = root or os.environ.get("IPP_DATASET_ROOT")
        if not root:
            raise ValueError(
                "Pass root=... or set IPP_DATASET_ROOT to the SROIE data folder "
                "(the one containing img/, box/, key/)."
            )
        self.root = Path(root)
        self.img_dir = self.root / "img"
        self.box_dir = self.root / "box"
        self.key_dir = self.root / "key"
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"No img/ under {self.root}")

    def ids(self) -> list[str]:
        return sorted(p.stem for p in self.img_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.ids())

    def load(self, sample_id: str) -> InvoiceSample:
        gt: dict[str, str] = {}
        key_path = self.key_dir / f"{sample_id}.json"
        if key_path.exists():
            raw = json.loads(key_path.read_text(encoding="utf-8"))
            for sroie_key, field_name in SROIE_TO_FIELD.items():
                if raw.get(sroie_key):
                    gt[field_name] = str(raw[sroie_key]).strip()

        boxes: list[GroundTruthBox] = []
        box_path = self.box_dir / f"{sample_id}.csv"
        if box_path.exists():
            boxes = self._read_boxes(box_path)

        return InvoiceSample(
            sample_id=sample_id,
            image_path=str(self.img_dir / f"{sample_id}.jpg"),
            ground_truth=gt,
            word_boxes=boxes,
        )

    def __iter__(self):
        for sid in self.ids():
            yield self.load(sid)

    @staticmethod
    def _read_boxes(path: Path) -> list[GroundTruthBox]:
        """Parse SROIE 8-coord polygons into axis-aligned boxes."""
        out: list[GroundTruthBox] = []
        with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
            for row in csv.reader(fh):
                if len(row) < 9:
                    continue
                try:
                    xs = [int(row[i]) for i in (0, 2, 4, 6)]
                    ys = [int(row[i]) for i in (1, 3, 5, 7)]
                except ValueError:
                    continue
                text = ",".join(row[8:]).strip()  # text may itself contain commas
                out.append(GroundTruthBox(text, (min(xs), min(ys), max(xs), max(ys))))
        return out


# --- scoring ----------------------------------------------------------------

def _norm_money(s: str) -> str:
    return "".join(c for c in s if c.isdigit() or c == ".")


def field_matches(field_name: str, predicted: str | None, expected: str) -> bool:
    """Lenient, field-aware comparison (OCR noise + formatting tolerant)."""
    if not predicted:
        return False
    p, e = predicted.strip().lower(), expected.strip().lower()
    if field_name == "total":
        return _norm_money(p) == _norm_money(e)
    if field_name == "supplier":
        # company names: compare on alphanumerics, allow substring either way
        pn = "".join(ch for ch in p if ch.isalnum())
        en = "".join(ch for ch in e if ch.isalnum())
        return bool(pn) and (pn in en or en in pn)
    return p.replace(" ", "") == e.replace(" ", "")


@dataclass
class FieldScore:
    field_name: str
    expected: str
    predicted: str | None
    correct: bool


def score_sample(predictions, ground_truth: dict[str, str]) -> list[FieldScore]:
    """Compare extractor predictions to a sample's ground truth."""
    pred_map = {p.field_name: p.value for p in predictions}
    scores: list[FieldScore] = []
    for field_name, expected in ground_truth.items():
        predicted = pred_map.get(field_name)
        scores.append(
            FieldScore(field_name, expected, predicted,
                       field_matches(field_name, predicted, expected))
        )
    return scores
