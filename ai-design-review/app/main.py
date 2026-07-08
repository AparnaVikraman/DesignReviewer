import json
import logging
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.document_service import DocumentService
from app.embedding_service import EmbeddingService
from app.errors import ReviewError, llm_error_handler, review_error_handler
from app.llm_client import LLMError
from app.metrics import get_metrics
from app.models import (
    CompareReviewResponse,
    DocumentUploadResponse,
    ReviewRequest,
    ReviewResponse,
)
from app.observability import log_review_event
from app.review_input import resolve_review_input
from app.review_service import compare_reviews, review_design, stream_review_design

load_dotenv()

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="AI Design Review Assistant")
app.add_exception_handler(ReviewError, review_error_handler)
app.add_exception_handler(LLMError, llm_error_handler)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    log_review_event(
        "http_request",
        request_id,
        method=request.method,
        path=request.url.path,
    )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health():
    metrics = get_metrics()
    return {
        "status": "ok",
        "reviews_total": metrics.reviews_total,
        "reviews_failed": metrics.reviews_failed,
    }


@app.get("/metrics")
def metrics():
    return get_metrics().to_dict()


@app.post("/documents", response_model=DocumentUploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    service = DocumentService()
    try:
        documents = service.upload_and_process_many(files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DocumentUploadResponse(documents=documents)


@app.post("/index")
def index_documents():
    count = EmbeddingService().index_documents()
    return {"status": "ok", "chunks_indexed": count}


@app.post("/review", response_model=ReviewResponse)
def review(request: ReviewRequest, http_request: Request):
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    resolved = resolve_review_input(request, request_id=request_id)
    return review_design(
        resolved.design_doc,
        use_retrieval=request.use_retrieval,
        source_document_id=resolved.source_document_id,
        source_filename=resolved.source_filename,
        request_id=request_id,
    )


@app.post("/review/compare", response_model=CompareReviewResponse)
def review_compare(request: ReviewRequest, http_request: Request):
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    resolved = resolve_review_input(request, request_id=request_id)
    return compare_reviews(
        resolved.design_doc,
        source_document_id=resolved.source_document_id,
        source_filename=resolved.source_filename,
    )


@app.post("/review/stream")
def review_stream(request: ReviewRequest, http_request: Request):
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    resolved = resolve_review_input(request, request_id=request_id)

    def event_generator():
        for stream_event in stream_review_design(
            resolved.design_doc,
            use_retrieval=request.use_retrieval,
            source_document_id=resolved.source_document_id,
            source_filename=resolved.source_filename,
            request_id=request_id,
        ):
            payload = json.dumps(stream_event.data)
            yield f"event: {stream_event.event}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )
