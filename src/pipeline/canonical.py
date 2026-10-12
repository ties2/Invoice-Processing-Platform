"""Value canonicalization (shared across all extractors).

Different extractors emit the same fact in different shapes: the rule-based one
writes money as "1000.00", an LLM may echo the page's "1000,00". Comparing raw
strings then reports a *formatting* difference as a semantic disagreement, which
pollutes the champion/challenger corner-case signal and the stored record.

Canonicalizing every prediction to one representation per field type fixes this at
the source: comparison, validation and storage all see the same normalized value.
Money -> plain decimal string; dates -> ISO (YYYY-MM-DD); currency -> ISO code;
VAT rate -> "NN%".
"""

from __future__ import annotations

import re
from datetime import datetime

from src.serving.db_models import FieldName

_MONEY_FIELDS = {
    FieldName.SUBTOTAL.value,
    FieldName.VAT_AMOUNT.value,
    FieldName.TOTAL.value,
}
_DATE_FORMATS = ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%y")
_CURRENCY_MAP = {"€": "EUR", "eur": "EUR", "$": "USD", "usd": "USD", "£": "GBP", "gbp": "GBP"}


def canonicalize_money(value: str) -> str:
    """'1.250,00' / '1,250.00' / '€ 1000,00' -> '1250.00'."""
    s = re.sub(r"[^\d.,]", "", value)
    if not s:
        return value
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):        # comma is the decimal sep
            s = s.replace(".", "").replace(",", ".")
        else:                                   # dot is the decimal sep
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if re.search(r",\d{2}$", s) else s.replace(",", "")
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return value


def canonicalize_date(value: str) -> str:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value


def canonicalize_currency(value: str) -> str:
    v = value.strip().lower()
    return _CURRENCY_MAP.get(v, value.strip().upper()[:3])


def canonicalize_rate(value: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)", value)
    return f"{m.group(1)}%" if m else value


def canonicalize(field_name: str, value: str | None) -> str | None:
    """Normalize one field's value to its canonical representation."""
    if value is None:
        return None
    value = value.strip()
    if field_name in _MONEY_FIELDS:
        return canonicalize_money(value)
    if field_name == FieldName.INVOICE_DATE.value:
        return canonicalize_date(value)
    if field_name == FieldName.CURRENCY.value:
        return canonicalize_currency(value)
    if field_name == FieldName.VAT_RATE.value:
        return canonicalize_rate(value)
    return value  # invoice_number, supplier: leave as-is
