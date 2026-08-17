from src.pipeline.ocr import SimulatedOCR
from src.pipeline.table_detector import SimulatedTableDetector

TABLE = (
    b"ACME Trading B.V.\n"
    b"Description        Qty     Price      Amount\n"
    b"Widget A          2       50,00      100,00\n"
    b"Cable set         10      12,50      125,00\n"
    b"Service fee       1       75,00      75,00\n"
    b"Subtotal                             300,00\n"
    b"Total                                363,00\n"
)


def test_detects_three_line_items_excluding_summary():
    ocr = SimulatedOCR(0.0).recognize(TABLE)
    res = SimulatedTableDetector().detect(ocr)
    assert res.n_items == 3
    descs = [i.description for i in res.line_items]
    assert "Subtotal" not in " ".join(descs)


def test_line_items_reconcile_to_subtotal():
    ocr = SimulatedOCR(0.0).recognize(TABLE)
    res = SimulatedTableDetector().detect(ocr)
    total = sum(float(i.amount.replace(",", ".")) for i in res.line_items)
    assert abs(total - 300.00) < 0.01


def test_no_table_returns_empty():
    ocr = SimulatedOCR(0.0).recognize(b"KPN B.V.\nTotal 100,00\n")
    assert SimulatedTableDetector().detect(ocr).n_items == 0
