"""Deterministic business validation (IPP-005).

The neural/rule extractor proposes values; this layer checks whether those values
are *financially coherent*, using arithmetic and format rules that owe nothing to
any model. This is the FinTech safety net: a high-confidence prediction that
fails `subtotal + VAT == total` must never be booked automatically.

Each rule yields zero or more `ValidationIssue`s with a severity. An ERROR blocks
auto-booking outright; WARNINGs lower trust and push borderline cases to review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from src.serving.db_models import FieldName


class Severity(str, Enum):
    ERROR = "error"      # financially incoherent -> block auto-book
    WARNING = "warning"  # suspicious -> lower trust / prefer review
    INFO = "info"


@dataclass
class ValidationIssue:
    rule: str
    severity: Severity
    message: str


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def is_valid(self) -> bool:
        return not self.has_errors


# --- parsing helpers --------------------------------------------------------
def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("€", "").strip())
    except ValueError:
        return None


def _rate_to_fraction(value: str | None) -> float | None:
    if value is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(m.group(1)) / 100.0 if m else None


def _parse_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%y"):
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


class InvoiceValidator:
    """Runs all financial validation rules over a field-name -> value mapping."""

    # Absolute tolerance for money comparisons (rounding noise).
    MONEY_TOLERANCE = 0.02

    def validate(self, fields: dict[str, str | None]) -> ValidationResult:
        result = ValidationResult()
        self._require_fields(fields, result)
        self._check_totals(fields, result)
        self._check_vat_amount(fields, result)
        self._check_date(fields, result)
        self._check_currency(fields, result)
        return result

    def _require_fields(self, f, result):
        for required in (FieldName.INVOICE_NUMBER.value, FieldName.TOTAL.value):
            if not f.get(required):
                result.issues.append(ValidationIssue(
                    rule="required_field",
                    severity=Severity.ERROR,
                    message=f"Missing required field: {required}",
                ))

    def _check_totals(self, f, result):
        """subtotal + vat_amount must equal total."""
        sub = _to_float(f.get(FieldName.SUBTOTAL.value))
        vat = _to_float(f.get(FieldName.VAT_AMOUNT.value))
        total = _to_float(f.get(FieldName.TOTAL.value))
        if sub is None or vat is None or total is None:
            return
        if abs((sub + vat) - total) > self.MONEY_TOLERANCE:
            result.issues.append(ValidationIssue(
                rule="totals_arithmetic",
                severity=Severity.ERROR,
                message=f"subtotal ({sub}) + vat ({vat}) = {sub + vat} != total ({total})",
            ))

    def _check_vat_amount(self, f, result):
        """vat_amount should be roughly subtotal * vat_rate."""
        sub = _to_float(f.get(FieldName.SUBTOTAL.value))
        rate = _rate_to_fraction(f.get(FieldName.VAT_RATE.value))
        vat = _to_float(f.get(FieldName.VAT_AMOUNT.value))
        if sub is None or rate is None or vat is None:
            return
        expected = round(sub * rate, 2)
        # A 1-cent-per-unit tolerance scaled to the amount.
        if abs(expected - vat) > max(self.MONEY_TOLERANCE, 0.01 * expected):
            result.issues.append(ValidationIssue(
                rule="vat_consistency",
                severity=Severity.WARNING,
                message=f"vat_amount {vat} != subtotal*rate ({expected})",
            ))

    def _check_date(self, f, result):
        raw = f.get(FieldName.INVOICE_DATE.value)
        if not raw:
            return
        parsed = _parse_date(raw)
        if parsed is None:
            result.issues.append(ValidationIssue(
                rule="date_format",
                severity=Severity.WARNING,
                message=f"Unparseable invoice date: {raw}",
            ))
            return
        now = datetime.now(UTC)
        if parsed > now:
            result.issues.append(ValidationIssue(
                rule="date_future",
                severity=Severity.WARNING,
                message=f"Invoice date is in the future: {raw}",
            ))

    def _check_currency(self, f, result):
        cur = f.get(FieldName.CURRENCY.value)
        if not cur:
            result.issues.append(ValidationIssue(
                rule="currency_missing",
                severity=Severity.WARNING,
                message="No currency detected",
            ))
