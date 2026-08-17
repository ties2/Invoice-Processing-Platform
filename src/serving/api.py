"""HTTP API (IPP-009).

Thin transport layer: every route validates input, delegates to the service, and
maps domain errors to HTTP status codes. No business logic lives here.

    POST /invoices                     upload a document        -> 201
    POST /invoices/{id}/process        run the pipeline         -> 200
    GET  /invoices/{id}                full detail + fields     -> 200
    GET  /invoices                     list (optional ?status)  -> 200
    POST /invoices/{id}/corrections    submit a human fix       -> 201
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.serving.analytics import AnalyticsService
from src.serving.database import get_db
from src.serving.dependency import get_artifact_storage
from src.serving.pipeline_factory import get_pipeline
from src.serving.repository import InvoiceRepository
from src.serving.schemas import (
    CorrectionCreate,
    CorrectionResponse,
    InvoiceDetailResponse,
    InvoiceResponse,
    ProcessingResult,
)
from src.serving.service import InvoiceNotFoundError, InvoiceService

router = APIRouter()


def get_service(db: Session = Depends(get_db)) -> InvoiceService:
    return InvoiceService(
        repository=InvoiceRepository(db),
        storage=get_artifact_storage(),
        pipeline=get_pipeline(),
    )


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/metrics")
def metrics(db: Session = Depends(get_db)):
    """Business KPIs computed live from the database."""
    return AnalyticsService(InvoiceRepository(db)).compute()


@router.post("/invoices", response_model=InvoiceResponse, status_code=201)
async def upload_invoice(
    file: UploadFile = File(...),
    service: InvoiceService = Depends(get_service),
):
    content = await file.read()
    invoice = service.register_invoice(filename=file.filename, content=content)
    return invoice


@router.post("/invoices/{invoice_id}/process", response_model=ProcessingResult)
def process_invoice(invoice_id: int, service: InvoiceService = Depends(get_service)):
    try:
        return service.process_invoice(invoice_id)
    except InvoiceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/invoices/{invoice_id}", response_model=InvoiceDetailResponse)
def get_invoice(invoice_id: int, service: InvoiceService = Depends(get_service)):
    try:
        return service.get_invoice(invoice_id)
    except InvoiceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/invoices", response_model=list[InvoiceResponse])
def list_invoices(status: str | None = None, service: InvoiceService = Depends(get_service)):
    return service.list_invoices(status=status)


@router.post(
    "/invoices/{invoice_id}/corrections",
    response_model=CorrectionResponse,
    status_code=201,
)
def submit_correction(
    invoice_id: int,
    body: CorrectionCreate,
    service: InvoiceService = Depends(get_service),
):
    try:
        return service.submit_correction(
            invoice_id=invoice_id,
            field_name=body.field_name,
            corrected_value=body.corrected_value,
            corrected_by=body.corrected_by,
        )
    except InvoiceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
