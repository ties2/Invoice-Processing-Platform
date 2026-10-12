from src.pipeline.canonical import canonicalize


def test_money_canonicalization():
    assert canonicalize("total", "1.250,00") == "1250.00"
    assert canonicalize("total", "1,250.00") == "1250.00"
    assert canonicalize("subtotal", "1000,00") == "1000.00"
    assert canonicalize("vat_amount", "€ 210,00") == "210.00"


def test_date_to_iso():
    assert canonicalize("invoice_date", "12-08-2026") == "2026-08-12"
    assert canonicalize("invoice_date", "2026-08-12") == "2026-08-12"


def test_currency_and_rate():
    assert canonicalize("currency", "€") == "EUR"
    assert canonicalize("currency", "eur") == "EUR"
    assert canonicalize("vat_rate", "21 %") == "21%"


def test_untouched_fields_passthrough():
    assert canonicalize("invoice_number", "INV-2026-1234") == "INV-2026-1234"
    assert canonicalize("supplier", "KPN B.V.") == "KPN B.V."
