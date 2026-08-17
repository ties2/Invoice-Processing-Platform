"""Service layer (IPP-009).

Business orchestration: it owns the *what* (register an invoice, process it, record
a human correction) and delegates the *how* to the pipeline (extraction), the
storage (artifact bytes) and the repository (persistence).

The three methods map onto the Zenvoices flow end to end:
    register_invoice  -> a document arrives and is stored
    process_invoice   -> OCR/extract/validate/route -> auto-book or needs-review
    submit_correction -> a human fixes a field -> feedback captured for retraining
"""

from __future__ import annotations

from src.pipeline.pipeline import ExtractionPipeline
from src.serving.db_models import Invoice, InvoiceStatus
from src.serving.repository import InvoiceRepository
from src.serving.schemas import ProcessingResult
from src.serving.storage import ArtifactStorage


class InvoiceNotFoundError(Exception):
    pass


class InvoiceService:
    def __init__(
        self,
        repository: InvoiceRepository,
        storage: ArtifactStorage,
        pipeline: ExtractionPipeline,
    ):
        self.repo = repository
        self.storage = storage
        self.pipeline = pipeline

    # --- 1. register ---------------------------------------------------------
    def register_invoice(self, filename: str, content: bytes) -> Invoice:
        """Store the raw document and create a RECEIVED invoice row."""
        invoice = self.repo.create(filename=filename, artifact_uri="pending")
        # URI namespaced by id keeps artifacts collision-free and traceable.
        uri = f"local://invoices/{invoice.id}/{filename}"
        self.storage.save(uri, content)
        self.repo.save_result(
            invoice_id=invoice.id,
            supplier=None,
            extractor_version="",
            overall_confidence=0.0,
            status=InvoiceStatus.RECEIVED,
        )
        # Persist the real artifact_uri now that we know the id.
        self.repo.db.query(Invoice).filter(Invoice.id == invoice.id).update(
            {"artifact_uri": uri}
        )
        self.repo.db.commit()
        return self.repo.get(invoice.id)

    # --- 2. process ----------------------------------------------------------
    def process_invoice(self, invoice_id: int) -> ProcessingResult:
        invoice = self.repo.get(invoice_id)
        if invoice is None:
            raise InvoiceNotFoundError(f"Invoice {invoice_id} not found")

        self.repo.set_status(invoice_id, InvoiceStatus.PROCESSING)

        try:
            content = self.storage.get(invoice.artifact_uri)
            outcome = self.pipeline.run(content, routing_key=invoice.artifact_uri)
        except Exception:
            # Any pipeline failure is a terminal state a human can inspect, never
            # a silent auto-book.
            self.repo.set_status(invoice_id, InvoiceStatus.FAILED)
            raise

        self.repo.replace_fields(invoice_id, outcome.predictions)
        self.repo.save_result(
            invoice_id=invoice_id,
            supplier=outcome.supplier,
            extractor_version=outcome.extractor_version,
            overall_confidence=outcome.decision.overall_confidence,
            status=outcome.decision.status,
        )

        return ProcessingResult(
            invoice_id=invoice_id,
            status=outcome.decision.status.value,
            overall_confidence=outcome.decision.overall_confidence,
            extractor_version=outcome.extractor_version,
            decision=outcome.decision.status.value,
            fields=[
                {
                    "field_name": p.field_name,
                    "field_value": p.value,
                    "confidence": p.confidence,
                    "source": p.source,
                }
                for p in outcome.predictions
            ],
        )

    # --- 3. correct (feedback loop) -----------------------------------------
    def submit_correction(
        self, invoice_id: int, field_name: str, corrected_value: str, corrected_by: str
    ):
        invoice = self.repo.get(invoice_id)
        if invoice is None:
            raise InvoiceNotFoundError(f"Invoice {invoice_id} not found")

        current = self.repo.get_field(invoice_id, field_name)
        original_value = current.field_value if current else None

        # 1) capture the append-only training signal
        correction = self.repo.add_correction(
            invoice_id=invoice_id,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            corrected_by=corrected_by,
        )
        # 2) apply the human's value to the live record
        if current:
            self.repo.update_field_value(invoice_id, field_name, corrected_value)
        # 3) a corrected invoice is a human-approved invoice
        self.repo.set_status(invoice_id, InvoiceStatus.APPROVED)
        return correction

    # --- reads ---------------------------------------------------------------
    def get_invoice(self, invoice_id: int) -> Invoice:
        invoice = self.repo.get(invoice_id)
        if invoice is None:
            raise InvoiceNotFoundError(f"Invoice {invoice_id} not found")
        return invoice

    def list_invoices(self, status: str | None = None):
        return self.repo.list(status=status)
