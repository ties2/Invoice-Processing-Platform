from src.pipeline.validation import InvoiceValidator, Severity


def test_valid_invoice_passes():
    r = InvoiceValidator().validate({
        "invoice_number": "INV-1", "subtotal": "1000.00",
        "vat_amount": "210.00", "total": "1210.00",
        "vat_rate": "21%", "invoice_date": "12-08-2026", "currency": "EUR",
    })
    assert r.is_valid


def test_arithmetic_mismatch_is_error():
    r = InvoiceValidator().validate({
        "invoice_number": "INV-1", "subtotal": "1000.00",
        "vat_amount": "210.00", "total": "9999.00", "currency": "EUR",
    })
    assert r.has_errors
    assert any(i.rule == "totals_arithmetic" for i in r.issues)


def test_missing_required_field_is_error():
    r = InvoiceValidator().validate({"subtotal": "10.00"})
    assert r.has_errors
    assert any(i.rule == "required_field" for i in r.issues)


def test_vat_inconsistency_is_warning():
    r = InvoiceValidator().validate({
        "invoice_number": "INV-1", "subtotal": "1000.00",
        "vat_rate": "21%", "vat_amount": "500.00", "total": "1500.00", "currency": "EUR",
    })
    assert any(i.rule == "vat_consistency" and i.severity == Severity.WARNING for i in r.issues)
