"""Pydantic v2 schemas (validation + serialization boundary).

Same convention as the reference project: separate `...Create` (what a client may
send) from `...Response` (what the server returns, including generated fields).
Response models use `from_attributes=True` so they serialize straight from ORM
objects.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# --- Extracted field ---
class ExtractedFieldResponse(BaseModel):
    field_name: str
    field_value: Optional[str] = None
    confidence: float
    source: str

    model_config = ConfigDict(from_attributes=True)


# --- Invoice ---
class InvoiceCreate(BaseModel):
    """Metadata a client sends when registering an invoice document.

    The raw file bytes are uploaded separately (multipart), the same way the
    reference project registered a model version and saved the artifact.
    """

    filename: str


class InvoiceResponse(BaseModel):
    id: int
    filename: str
    supplier_name: Optional[str] = None
    extractor_version: Optional[str] = None
    overall_confidence: Optional[float] = None
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InvoiceDetailResponse(InvoiceResponse):
    """Full view: the invoice plus every extracted field."""

    fields: list[ExtractedFieldResponse] = []

    model_config = ConfigDict(from_attributes=True)


# --- Processing result (returned right after the pipeline runs) ---
class ProcessingResult(BaseModel):
    invoice_id: int
    status: str
    overall_confidence: float
    extractor_version: str
    decision: str  # "auto_booked" | "needs_review"
    fields: list[ExtractedFieldResponse] = []


# --- Human correction (feedback loop input) ---
class CorrectionCreate(BaseModel):
    field_name: str
    corrected_value: str
    corrected_by: str = "unknown"


class CorrectionResponse(BaseModel):
    id: int
    invoice_id: int
    field_name: str
    original_value: Optional[str] = None
    corrected_value: Optional[str] = None
    corrected_by: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
