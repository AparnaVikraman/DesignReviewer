import logging
from dataclasses import dataclass

from fastapi import HTTPException

from app.document_service import DocumentService
from app.errors import ReviewError
from app.models import ReviewRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedReviewInput:
    design_doc: str
    source_document_id: str | None = None
    source_filename: str | None = None


def resolve_review_input(
    request: ReviewRequest,
    *,
    request_id: str,
) -> ResolvedReviewInput:
    if request.design_doc is not None:
        design_doc = request.design_doc.strip()
        if not design_doc:
            raise ReviewError(
                "Design document is empty",
                error_type="empty_design_doc",
                request_id=request_id,
                retryable=False,
                status_code=400,
            )
        return ResolvedReviewInput(design_doc=design_doc)

    service = DocumentService()
    try:
        design_doc, document = service.read_design_text(
            document_id=request.document_id,
            filename=request.filename,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise ReviewError(
            str(exc),
            error_type="invalid_document",
            request_id=request_id,
            retryable=False,
            status_code=400,
        ) from exc

    if not design_doc.strip():
        raise ReviewError(
            "Uploaded design document contains no readable text",
            error_type="empty_design_doc",
            request_id=request_id,
            retryable=False,
            status_code=400,
        )

    logger.info(
        "Resolved review input request_id=%s document_id=%s filename=%s",
        request_id,
        document.id,
        document.filename,
    )
    return ResolvedReviewInput(
        design_doc=design_doc,
        source_document_id=document.id,
        source_filename=document.filename,
    )
