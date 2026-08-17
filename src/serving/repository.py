"""Data-access layer (IPP-009).

All database reads/writes live here, behind method calls the service can use
without ever touching a SQLAlchemy query — the same repository pattern as the
reference project. The service owns *what* to do; the repository owns *how* it is
persisted.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.pipeline.extractor import FieldPrediction
from src.serving.db_models import Correction, ExtractedField, Invoice, InvoiceStatus


class InvoiceRepository:
    def __init__(self, db: Session):
        self.db = db

    # --- invoices ------------------------------------------------------------
    def create(self, filename: str, artifact_uri: str) -> Invoice:
        invoice = Invoice(
            filename=filename,
            artifact_uri=artifact_uri,
            status=InvoiceStatus.RECEIVED.value,
        )
        self.db.add(invoice)
        self.db.commit()
        self.db.refresh(invoice)
        return invoice

    def get(self, invoice_id: int) -> Invoice | None:
        return self.db.query(Invoice).filter(Invoice.id == invoice_id).first()

    def list(self, status: str | None = None, limit: int = 50, offset: int = 0) -> list[Invoice]:
        q = self.db.query(Invoice)
        if status:
            q = q.filter(Invoice.status == status)
        return q.order_by(Invoice.id.desc()).offset(offset).limit(limit).all()

    def set_status(self, invoice_id: int, status: InvoiceStatus) -> None:
        self.db.query(Invoice).filter(Invoice.id == invoice_id).update(
            {"status": status.value}
        )
        self.db.commit()

    def save_result(
        self,
        invoice_id: int,
        supplier: str | None,
        extractor_version: str,
        overall_confidence: float,
        status: InvoiceStatus,
    ) -> None:
        self.db.query(Invoice).filter(Invoice.id == invoice_id).update({
            "supplier_name": supplier,
            "extractor_version": extractor_version,
            "overall_confidence": overall_confidence,
            "status": status.value,
        })
        self.db.commit()

    # --- extracted fields ----------------------------------------------------
    def replace_fields(self, invoice_id: int, predictions: list[FieldPrediction]) -> None:
        """Overwrite the current field record for an invoice (idempotent reprocess)."""
        self.db.query(ExtractedField).filter(
            ExtractedField.invoice_id == invoice_id
        ).delete()
        for p in predictions:
            self.db.add(ExtractedField(
                invoice_id=invoice_id,
                field_name=p.field_name,
                field_value=p.value,
                confidence=p.confidence,
                source=p.source,
            ))
        self.db.commit()

    def get_field(self, invoice_id: int, field_name: str) -> ExtractedField | None:
        return (
            self.db.query(ExtractedField)
            .filter(
                ExtractedField.invoice_id == invoice_id,
                ExtractedField.field_name == field_name,
            )
            .first()
        )

    def update_field_value(self, invoice_id: int, field_name: str, value: str) -> None:
        self.db.query(ExtractedField).filter(
            ExtractedField.invoice_id == invoice_id,
            ExtractedField.field_name == field_name,
        ).update({"field_value": value})
        self.db.commit()

    # --- corrections (feedback loop) -----------------------------------------
    def add_correction(
        self,
        invoice_id: int,
        field_name: str,
        original_value: str | None,
        corrected_value: str,
        corrected_by: str,
    ) -> Correction:
        correction = Correction(
            invoice_id=invoice_id,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            corrected_by=corrected_by,
        )
        self.db.add(correction)
        self.db.commit()
        self.db.refresh(correction)
        return correction

    def list_corrections(self, invoice_id: int | None = None) -> list[Correction]:
        q = self.db.query(Correction)
        if invoice_id is not None:
            q = q.filter(Correction.invoice_id == invoice_id)
        return q.order_by(Correction.id.desc()).all()

    # --- aggregates (analytics / monitoring) ---------------------------------
    def count_by_status(self) -> dict[str, int]:
        from sqlalchemy import func
        rows = (
            self.db.query(Invoice.status, func.count(Invoice.id))
            .group_by(Invoice.status)
            .all()
        )
        return {status: count for status, count in rows}

    def confidences(self) -> list[float]:
        rows = (
            self.db.query(Invoice.overall_confidence)
            .filter(Invoice.overall_confidence.isnot(None))
            .filter(Invoice.overall_confidence > 0)
            .all()
        )
        return [r[0] for r in rows]

    def correction_field_counts(self) -> dict[str, int]:
        from sqlalchemy import func
        rows = (
            self.db.query(Correction.field_name, func.count(Correction.id))
            .group_by(Correction.field_name)
            .all()
        )
        return {field: count for field, count in rows}

    def count_invoices_with_corrections(self) -> int:
        from sqlalchemy import func
        return (
            self.db.query(func.count(func.distinct(Correction.invoice_id))).scalar() or 0
        )
