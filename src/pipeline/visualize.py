"""Result visualisation (IPP-010).

Renders what the pipeline *found* back onto the real page image, side by side
with the original — the document-AI equivalent of overlaying a segmentation
mask on the source. Left: the raw invoice/receipt image. Right: the same image
with the OCR word boxes (faint), the extracted header fields highlighted and
labelled, and the detected line-item table columns shaded.

Why this lives behind its own module: the field extractor returns values and
confidences, not geometry. To draw a field we have to *relocate* its value in
the OCR word boxes. That matching is the only non-trivial part and it is kept
here so the extractor stays about extraction and the detector about detection.

Dependency note: matplotlib + Pillow only. This module duck-types the pipeline
objects (`ocr.words[*].bbox/.text`, `pred.field_name/.value/.confidence`,
`table.column_bounds/.line_items`) so importing it never pulls in the DB layer.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass


# --- field -> colour, drawn in a fixed order so the legend is stable ---------
FIELD_STYLE = {
    "supplier":       ("#2563eb", "Supplier"),
    "invoice_number": ("#16a34a", "Invoice no."),
    "invoice_date":   ("#9333ea", "Date"),
    "vat_rate":       ("#0891b2", "VAT rate"),
    "vat_amount":     ("#0891b2", "VAT amount"),
    "subtotal":       ("#d97706", "Subtotal"),
    "total":          ("#dc2626", "Total"),
    "currency":       ("#64748b", "Currency"),
}

BBox = tuple[int, int, int, int]  # (x0, y0, x1, y1)


@dataclass
class LocatedField:
    """An extracted field re-anchored to a box on the page."""
    field_name: str
    value: str
    confidence: float
    bbox: BBox | None  # None when the value could not be relocated


# --- value <-> OCR matching --------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase and drop everything but letters/digits, so '12/03/18',
    'RM77.20' and '77,20' compare on their meaningful characters only."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def locate_value(words, value: str, max_window: int = 8) -> BBox | None:
    """Find the span of OCR words whose joined text matches `value`.

    Slides a window of 1..max_window consecutive words and keeps the *smallest*
    window whose normalised text contains (or is contained by) the normalised
    value. Returns the union bounding box of that span, or None.
    """
    target = _norm(value)
    if not target or not words:
        return None

    best: tuple[int, BBox] | None = None  # (span_len, bbox)
    n = len(words)
    for i in range(n):
        joined = ""
        for w in range(i, min(i + max_window, n)):
            joined += _norm(words[w].text)
            if not joined:
                continue
            hit = target in joined or (len(joined) >= 3 and joined in target)
            if hit:
                span = words[i : w + 1]
                x0 = min(s.bbox[0] for s in span)
                y0 = min(s.bbox[1] for s in span)
                x1 = max(s.bbox[2] for s in span)
                y1 = max(s.bbox[3] for s in span)
                span_len = w - i + 1
                if best is None or span_len < best[0]:
                    best = (span_len, (x0, y0, x1, y1))
                break  # window from i already matched; shrink by moving i
    return best[1] if best else None


def locate_fields(ocr, predictions) -> list[LocatedField]:
    """Relocate every non-empty prediction to a box on the page."""
    located: list[LocatedField] = []
    for p in predictions:
        if not p.value:
            continue
        located.append(
            LocatedField(p.field_name, p.value, p.confidence,
                         locate_value(ocr.words, p.value))
        )
    return located


# --- rendering ---------------------------------------------------------------

def render_result(
    image_bytes: bytes,
    ocr,
    predictions,
    table=None,
    *,
    title: str = "Invoice Processing Platform — extraction result",
    decision: str | None = None,
    save_path: str | None = None,
    dpi: int = 130,
):
    """Draw original vs. overlay and return the matplotlib Figure.

    `decision` is an optional routing outcome string (e.g. 'auto_booked').
    If `save_path` is given the figure is written there too.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    located = locate_fields(ocr, predictions)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 8))
    for ax in (axL, axR):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])

    axL.set_title("Original page", fontsize=11)
    axR.set_title("Detected fields + OCR boxes", fontsize=11)

    # faint OCR word boxes = "everything OCR saw"
    for w in ocr.words:
        x0, y0, x1, y1 = w.bbox
        axR.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                fill=False, edgecolor="#9ca3af",
                                linewidth=0.4, alpha=0.5))

    # detected line-item columns = shaded bands (the "table mask")
    if table is not None and getattr(table, "column_bounds", None):
        ys = [y1 for *_, y1 in (w.bbox for w in ocr.words)] or [img.height]
        for col, (cx0, cx1) in table.column_bounds.items():
            axR.add_patch(Rectangle((cx0, 0), cx1 - cx0, max(ys),
                                    fill=True, facecolor="#f59e0b",
                                    alpha=0.07, edgecolor="none"))

    # extracted header fields = bold coloured boxes + labels
    used = set()
    for lf in located:
        colour, label = FIELD_STYLE.get(lf.field_name, ("#111827", lf.field_name))
        used.add(lf.field_name)
        if lf.bbox is None:
            continue
        x0, y0, x1, y1 = lf.bbox
        axR.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                fill=False, edgecolor=colour, linewidth=2.2))
        axR.text(x0, max(0, y0 - 4), f"{label}: {lf.value} ({lf.confidence:.2f})",
                 fontsize=7, color="white", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.15", fc=colour, ec="none", alpha=0.9))

    legend = [Patch(facecolor=FIELD_STYLE[f][0], label=FIELD_STYLE[f][1])
              for f in FIELD_STYLE if f in used]
    if table is not None and getattr(table, "column_bounds", None):
        legend.append(Patch(facecolor="#f59e0b", alpha=0.3, label="Line-item columns"))
    if legend:
        axR.legend(handles=legend, loc="upper right", fontsize=6.5,
                   framealpha=0.9, borderpad=0.6)

    sub = title
    if decision:
        sub += f"    |    routing decision: {decision}"
    fig.suptitle(sub, fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
