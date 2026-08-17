from src.pipeline.extractor import FieldPrediction
from src.pipeline.routing import RoutingEngine, aggregate_confidence
from src.pipeline.validation import InvoiceValidator, ValidationResult, ValidationIssue, Severity
from src.serving.db_models import InvoiceStatus


def preds(conf):
    return [FieldPrediction("total", "1210.00", conf), FieldPrediction("invoice_number", "INV-1", conf)]


def test_aggregate_is_minimum():
    p = [FieldPrediction("a", "x", 0.99), FieldPrediction("b", "y", 0.60)]
    assert aggregate_confidence(p) == 0.60


def test_high_confidence_auto_books():
    d = RoutingEngine().decide(preds(0.98), ValidationResult(), is_corner_case=False)
    assert d.status == InvoiceStatus.AUTO_BOOKED


def test_low_confidence_routes_to_review():
    d = RoutingEngine().decide(preds(0.50), ValidationResult(), is_corner_case=False)
    assert d.status == InvoiceStatus.NEEDS_REVIEW


def test_validation_error_blocks_autobook_even_at_high_confidence():
    v = ValidationResult(issues=[ValidationIssue("totals_arithmetic", Severity.ERROR, "x")])
    d = RoutingEngine().decide(preds(0.99), v, is_corner_case=False)
    assert d.status == InvoiceStatus.NEEDS_REVIEW
    assert "validation_error" in d.reasons


def test_corner_case_blocks_autobook():
    d = RoutingEngine().decide(preds(0.99), ValidationResult(), is_corner_case=True)
    assert d.status == InvoiceStatus.NEEDS_REVIEW
    assert "corner_case" in d.reasons
