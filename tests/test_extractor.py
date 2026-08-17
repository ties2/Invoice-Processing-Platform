from src.pipeline.ocr import SimulatedOCR
from src.pipeline.extractor import RuleBasedExtractor


def _extract(raw, known=None):
    ocr = SimulatedOCR(noise=0.0).recognize(raw)
    return {p.field_name: p for p in RuleBasedExtractor(known_suppliers=known or set()).extract(ocr)}


def test_money_normalisation():
    ex = RuleBasedExtractor()
    assert ex._normalize_money("1.250,00") == "1250.00"
    assert ex._normalize_money("1,250.00") == "1250.00"
    assert ex._normalize_money("1000,00") == "1000.00"


def test_amounts_are_line_anchored(clean_invoice_bytes):
    f = _extract(clean_invoice_bytes)
    # The classic bug: subtotal != total, and each amount comes from its own line.
    assert f["subtotal"].value == "1000.00"
    assert f["vat_amount"].value == "210.00"
    assert f["total"].value == "1210.00"


def test_known_supplier_scores_higher_than_unknown(clean_invoice_bytes):
    known = _extract(clean_invoice_bytes, known={"KPN B.V."})["supplier"]
    unknown = _extract(clean_invoice_bytes)["supplier"]
    assert known.confidence > unknown.confidence
    assert known.source == "registry"


def test_invoice_number_and_currency(clean_invoice_bytes):
    f = _extract(clean_invoice_bytes)
    assert f["invoice_number"].value == "INV-2026-1234"
    assert f["currency"].value == "EUR"
