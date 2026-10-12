"""Run the real pipeline on real SROIE receipts and visualise the result.

    # 1. get the dataset (see src/datasets/sroie.py for sources), then:
    export IPP_DATASET_ROOT=/path/to/SROIE/data      # contains img/ box/ key/
    pip install -r requirements-ml.txt               # tesseract + pillow + matplotlib

    # one receipt -> side-by-side figure (original vs detected fields)
    python scripts/demo_visualize.py --id 060 --out result_060.png

    # scan N receipts and print per-field extraction accuracy vs ground truth
    python scripts/demo_visualize.py --eval 50

Everything here uses the production stages unchanged: TesseractOCR ->
RuleBasedExtractor -> SimulatedTableDetector -> InvoiceValidator ->
RoutingEngine. Only the *input* (real image instead of a text blob) and the
*output* (an overlay image instead of JSON) are new.
"""

from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse

from src.datasets.sroie import SroieDataset, score_sample
from src.pipeline.ocr_tesseract import TesseractOCR
from src.pipeline.extractor import RuleBasedExtractor
from src.pipeline.table_detector import SimulatedTableDetector
from src.pipeline.validation import InvoiceValidator
from src.pipeline.routing import RoutingEngine
from src.pipeline.visualize import render_result


def run_one(sample, ocr_engine, extractor, detector):
    image_bytes = sample.image_bytes
    ocr = ocr_engine.recognize(image_bytes)
    preds = extractor.extract(ocr)
    table = detector.detect(ocr)
    fields = {p.field_name: p.value for p in preds}
    vres = InvoiceValidator().validate(fields)
    decision = RoutingEngine().decide(preds, vres)
    return image_bytes, ocr, preds, table, decision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="SROIE data folder (or IPP_DATASET_ROOT)")
    ap.add_argument("--id", default=None, help="sample id to visualise, e.g. 060")
    ap.add_argument("--out", default="result.png", help="output PNG path")
    ap.add_argument("--eval", type=int, default=0, help="score accuracy over the first N samples")
    ap.add_argument("--lang", default="eng", help="Tesseract language(s)")
    args = ap.parse_args()

    ds = SroieDataset(args.root)
    ocr_engine = TesseractOCR(lang=args.lang)
    extractor = RuleBasedExtractor()
    detector = SimulatedTableDetector()

    if args.eval:
        hits: dict[str, int] = {}
        totals: dict[str, int] = {}
        for sid in ds.ids()[: args.eval]:
            sample = ds.load(sid)
            _, _, preds, *_ = run_one(sample, ocr_engine, extractor, detector)
            for s in score_sample(preds, sample.ground_truth):
                totals[s.field_name] = totals.get(s.field_name, 0) + 1
                hits[s.field_name] = hits.get(s.field_name, 0) + int(s.correct)
        print(f"\nExtraction accuracy over {args.eval} real receipts:")
        for f in sorted(totals):
            print(f"  {f:16s} {hits[f]:3d}/{totals[f]:<3d} = {hits[f]/totals[f]:.0%}")
        return

    sid = args.id or ds.ids()[0]
    sample = ds.load(sid)
    image_bytes, ocr, preds, table, decision = run_one(
        sample, ocr_engine, extractor, detector
    )
    render_result(
        image_bytes, ocr, preds, table,
        decision=decision.status.value, save_path=args.out,
    )
    print(f"receipt {sid}: {decision.status.value} (conf {decision.overall_confidence})")
    for s in score_sample(preds, sample.ground_truth):
        mark = "OK " if s.correct else "XX "
        print(f"  {mark}{s.field_name:16s} pred={s.predicted!r:40s} gt={s.expected!r}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
