"""Database tables (SQLAlchemy ORM).

Domain model for an invoice-processing platform. Where the reference MSP project
had `models` + `model_versions`, here the central entity is the **invoice**
flowing through a pipeline, with its **extracted fields** and the **human
corrections** that feed the learning loop.

    invoices  1───N  extracted_fields
        │
        └────1───N  corrections   (human-in-the-loop feedback)

Design note: the reference project flagged "status is free text" as a known
limitation. Here we harden it: statuses come from a Python Enum and are pinned
at the database level with a CHECK constraint, so an invalid status cannot be
written even by a raw SQL insert.
"""

import enum
from datetime import UTC, datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from src.serving.database import Base


class InvoiceStatus(str, enum.Enum):
    """Lifecycle of an invoice as it moves through the pipeline."""

    RECEIVED = "received"          # uploaded, not processed yet
    PROCESSING = "processing"      # pipeline running
    NEEDS_REVIEW = "needs_review"  # low/medium confidence -> human must check
    AUTO_BOOKED = "auto_booked"    # high confidence -> booked without a human
    APPROVED = "approved"          # a human confirmed/corrected and approved it
    REJECTED = "rejected"          # a human rejected it
    FAILED = "failed"              # pipeline error (bad OCR, corrupt file, ...)


# Canonical set of fields we try to extract from every invoice.
# Kept as constants so the extractor, validator and API all agree on names.
class FieldName(str, enum.Enum):
    INVOICE_NUMBER = "invoice_number"
    SUPPLIER = "supplier"
    INVOICE_DATE = "invoice_date"
    VAT_RATE = "vat_rate"
    VAT_AMOUNT = "vat_amount"
    SUBTOTAL = "subtotal"
    TOTAL = "total"
    CURRENCY = "currency"


class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)

    # Where the raw PDF/image bytes are stored (via the storage abstraction).
    artifact_uri = Column(String, nullable=False)

    # Filled in after extraction (nullable until the pipeline runs).
    supplier_name = Column(String, nullable=True)

    # Which extractor version produced the result that is currently on record.
    # This is our equivalent of "which model version served this request".
    extractor_version = Column(String, nullable=True)

    # Aggregate confidence across all fields (drives the routing decision).
    overall_confidence = Column(Float, nullable=True)

    status = Column(
        String,
        nullable=False,
        default=InvoiceStatus.RECEIVED.value,
        index=True,
    )

    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    fields = relationship(
        "ExtractedField",
        back_populates="invoice",
        cascade="all, delete-orphan",
    )
    corrections = relationship(
        "Correction",
        back_populates="invoice",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        # Harden the status column at the DB level (fixes the reference project's
        # "status is free text" limitation).
        CheckConstraint(
            "status IN ('received','processing','needs_review',"
            "'auto_booked','approved','rejected','failed')",
            name="ck_invoice_status",
        ),
    )


class ExtractedField(Base):
    """One extracted field for one invoice, with its own confidence score.

    We store confidence *per field*, not just per document, because that is what
    lets us route intelligently (e.g. total is uncertain even though supplier is
    certain) and because it is the unit a human corrects.
    """

    __tablename__ = "extracted_fields"

    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)

    field_name = Column(String, nullable=False)     # e.g. "total"
    field_value = Column(String, nullable=True)      # stored as text, typed on read
    confidence = Column(Float, nullable=False, default=0.0)

    # Which stage produced this value: useful for error analysis and drift.
    # e.g. "regex", "layout_model", "llm".
    source = Column(String, nullable=False, default="regex")

    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    invoice = relationship("Invoice", back_populates="fields")

    __table_args__ = (
        # A given field appears at most once per invoice on the current record.
        Index("uix_invoice_field", "invoice_id", "field_name", unique=True),
    )


class Correction(Base):
    """A human correction of one field — the raw material of the feedback loop.

    Every correction is an append-only training signal: it captures what the
    model predicted, what the human changed it to, and who changed it. A future
    retraining job consumes this table; nothing is ever overwritten here.
    """

    __tablename__ = "corrections"

    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)

    field_name = Column(String, nullable=False)
    original_value = Column(String, nullable=True)   # what the model predicted
    corrected_value = Column(String, nullable=True)  # what the human entered
    corrected_by = Column(String, nullable=False, default="unknown")

    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    invoice = relationship("Invoice", back_populates="corrections")
