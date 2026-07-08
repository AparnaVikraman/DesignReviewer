import logging
import time
import uuid
from collections.abc import Iterator

from app.config import get_settings
from app.errors import ReviewError
from app.llm_client import LLMClient, LLMError, StreamMetadata, estimate_cost_usd, usage_to_token_counts
from app.metrics import record_review_failure, record_review_success
from app.models import (
    CompareReviewResponse,
    DesignReview,
    LatencyBreakdown,
    RetrievedChunk,
    ReviewMetadata,
    ReviewResponse,
    StreamEvent,
    TokenUsage,
    design_review_fallback,
    openai_json_schema,
)
from app.observability import log_retrieved_documents, log_review_event
from app.prompts import PromptBuilder
from app.retriever import Retriever

logger = logging.getLogger(__name__)

_client: LLMClient | None = None
_retriever: Retriever | None = None
_prompt_builder: PromptBuilder | None = None
DESIGN_REVIEW_SCHEMA = openai_json_schema(DesignReview)
JSON_SCHEMA_NAME = "design_review"


def get_client() -> LLMClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = LLMClient(
            model=settings.model_name,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )
    return _client


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_prompt_builder() -> PromptBuilder:
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder


def _validate_design_doc(design_doc: str, request_id: str) -> None:
    if not design_doc.strip():
        raise ReviewError(
            "Design document is empty",
            error_type="empty_design_doc",
            request_id=request_id,
            retryable=False,
            status_code=400,
        )


def _build_metadata(
    *,
    model: str,
    latency: LatencyBreakdown,
    usage: object | None,
    retrieval_enabled: bool,
    retrieved_chunks: list[RetrievedChunk],
    source_document_id: str | None = None,
    source_filename: str | None = None,
) -> ReviewMetadata:
    input_tokens, output_tokens, total_tokens = usage_to_token_counts(usage)
    token_usage = None
    estimated_cost_usd = None
    if total_tokens > 0:
        token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        estimated_cost_usd = estimate_cost_usd(model, input_tokens, output_tokens)

    return ReviewMetadata(
        model=model,
        latency_ms=latency.total_ms,
        latency=latency,
        token_usage=token_usage,
        estimated_cost_usd=estimated_cost_usd,
        retrieval_enabled=retrieval_enabled,
        retrieved_chunks=retrieved_chunks,
        source_document_id=source_document_id,
        source_filename=source_filename,
    )


def _metadata_from_stream(
    stream_meta: StreamMetadata,
    latency: LatencyBreakdown,
    *,
    retrieval_enabled: bool,
    retrieved_chunks: list[RetrievedChunk],
    source_document_id: str | None = None,
    source_filename: str | None = None,
) -> ReviewMetadata:
    token_usage = None
    estimated_cost_usd = None
    if stream_meta.total_tokens > 0:
        token_usage = TokenUsage(
            input_tokens=stream_meta.input_tokens,
            output_tokens=stream_meta.output_tokens,
            total_tokens=stream_meta.total_tokens,
        )
        estimated_cost_usd = estimate_cost_usd(
            stream_meta.model,
            stream_meta.input_tokens,
            stream_meta.output_tokens,
        )
    return ReviewMetadata(
        model=stream_meta.model,
        latency_ms=latency.total_ms,
        latency=latency,
        token_usage=token_usage,
        estimated_cost_usd=estimated_cost_usd,
        retrieval_enabled=retrieval_enabled,
        retrieved_chunks=retrieved_chunks,
        source_document_id=source_document_id,
        source_filename=source_filename,
    )


def _build_prompt(
    design_doc: str,
    use_retrieval: bool,
    *,
    exclude_document_id: str | None = None,
) -> tuple[str, list[RetrievedChunk], float | None, float | None]:
    retrieved_chunks: list[RetrievedChunk] = []
    embedding_ms: float | None = None
    retrieval_ms: float | None = None
    prompt_builder = get_prompt_builder()

    if use_retrieval:
        result = get_retriever().retrieve(
            design_doc,
            exclude_document_id=exclude_document_id,
        )
        retrieved_chunks = result.chunks
        embedding_ms = result.embedding_ms
        retrieval_ms = result.search_ms
        prompt = prompt_builder.build(design_doc, retrieved_chunks)
    else:
        prompt = prompt_builder.build(design_doc)

    return prompt, retrieved_chunks, embedding_ms, retrieval_ms


def _log_completion(
    *,
    request_id: str,
    metadata: ReviewMetadata,
    validation_path: str | None,
    use_retrieval: bool,
) -> None:
    log_retrieved_documents(
        request_id,
        metadata.retrieved_chunks,
        retrieval_enabled=use_retrieval,
    )
    log_review_event(
        "review_complete",
        request_id,
        model=metadata.model,
        validation_path=validation_path,
        total_ms=metadata.latency_ms,
        embedding_ms=metadata.latency.embedding_ms if metadata.latency else None,
        retrieval_ms=metadata.latency.retrieval_ms if metadata.latency else None,
        llm_ms=metadata.latency.llm_ms if metadata.latency else None,
        input_tokens=metadata.token_usage.input_tokens if metadata.token_usage else 0,
        output_tokens=metadata.token_usage.output_tokens if metadata.token_usage else 0,
        cost_usd=metadata.estimated_cost_usd,
        retrieved_chunks=len(metadata.retrieved_chunks),
    )


def review_design(
    design_doc: str,
    *,
    use_retrieval: bool = True,
    source_document_id: str | None = None,
    source_filename: str | None = None,
    request_id: str | None = None,
) -> ReviewResponse:
    request_id = request_id or str(uuid.uuid4())
    _validate_design_doc(design_doc, request_id)
    log_review_event(
        "request_received",
        request_id,
        retrieval_enabled=use_retrieval,
        source_filename=source_filename,
    )

    total_start = time.perf_counter()
    try:
        prompt, retrieved_chunks, embedding_ms, retrieval_ms = _build_prompt(
            design_doc,
            use_retrieval,
            exclude_document_id=source_document_id,
        )

        llm = get_client()
        llm_start = time.perf_counter()
        log_review_event("model_called", request_id, model=llm.model)

        result = llm.complete(
            prompt,
            request_id=request_id,
            json_schema=DESIGN_REVIEW_SCHEMA,
            json_schema_name=JSON_SCHEMA_NAME,
            response_model=DesignReview,
            fallback=lambda _error: design_review_fallback(),
        )
        llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)

        review = result.validated
        if not isinstance(review, DesignReview):
            raise LLMError(
                "Validated review missing from LLM response",
                error_type="bad_response",
                request_id=request_id,
                retryable=False,
            )

        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        latency = LatencyBreakdown(
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        )
        metadata = _build_metadata(
            model=result.model,
            latency=latency,
            usage=result.usage,
            retrieval_enabled=use_retrieval,
            retrieved_chunks=retrieved_chunks,
            source_document_id=source_document_id,
            source_filename=source_filename,
        )
        input_tokens, output_tokens, _ = usage_to_token_counts(result.usage)
        record_review_success(
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=metadata.estimated_cost_usd or 0.0,
            retrieved_chunk_count=len(retrieved_chunks),
        )
        _log_completion(
            request_id=result.request_id,
            metadata=metadata,
            validation_path=result.validation_path,
            use_retrieval=use_retrieval,
        )

        return ReviewResponse(
            review=review,
            request_id=result.request_id,
            validation_path=result.validation_path,
            metadata=metadata,
        )
    except (LLMError, ReviewError):
        record_review_failure()
        raise
    except Exception:
        record_review_failure()
        raise


def compare_reviews(
    design_doc: str,
    *,
    source_document_id: str | None = None,
    source_filename: str | None = None,
) -> CompareReviewResponse:
    log_review_event("compare_started", str(uuid.uuid4()))
    without_retrieval = review_design(
        design_doc,
        use_retrieval=False,
        source_document_id=source_document_id,
        source_filename=source_filename,
    )
    with_retrieval = review_design(
        design_doc,
        use_retrieval=True,
        source_document_id=source_document_id,
        source_filename=source_filename,
    )
    return CompareReviewResponse(
        design_doc=design_doc,
        without_retrieval=without_retrieval,
        with_retrieval=with_retrieval,
    )


def stream_review_design(
    design_doc: str,
    *,
    use_retrieval: bool = True,
    source_document_id: str | None = None,
    source_filename: str | None = None,
    request_id: str | None = None,
) -> Iterator[StreamEvent]:
    request_id = request_id or str(uuid.uuid4())
    _validate_design_doc(design_doc, request_id)
    log_review_event(
        "request_received",
        request_id,
        stream=True,
        retrieval_enabled=use_retrieval,
        source_filename=source_filename,
    )

    total_start = time.perf_counter()
    embedding_ms: float | None = None
    retrieval_ms: float | None = None
    retrieved_chunks: list[RetrievedChunk] = []

    try:
        prompt, retrieved_chunks, embedding_ms, retrieval_ms = _build_prompt(
            design_doc,
            use_retrieval,
            exclude_document_id=source_document_id,
        )
        chunks: list[str] = []
        stream_meta: StreamMetadata | None = None

        llm = get_client()
        llm_start = time.perf_counter()
        log_review_event("model_called", request_id, model=llm.model, stream=True)

        for item in llm.complete_stream(
            prompt,
            request_id=request_id,
            json_schema=DESIGN_REVIEW_SCHEMA,
            json_schema_name=JSON_SCHEMA_NAME,
        ):
            if isinstance(item, StreamMetadata):
                stream_meta = item
                continue
            chunks.append(item)
            yield StreamEvent(event="delta", data={"text": item})

        llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)
        full_text = "".join(chunks)
        review, _, validation_path = llm.finalize_structured_response(
            full_text,
            request_id=request_id,
            response_model=DesignReview,
            json_schema=DESIGN_REVIEW_SCHEMA,
            json_schema_name=JSON_SCHEMA_NAME,
            fallback=lambda _error: design_review_fallback(),
        )

        if not isinstance(review, DesignReview):
            raise LLMError(
                "Validated review missing from streamed LLM response",
                error_type="bad_response",
                request_id=request_id,
                retryable=False,
            )

        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        latency = LatencyBreakdown(
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        )
        metadata = (
            _metadata_from_stream(
                stream_meta,
                latency,
                retrieval_enabled=use_retrieval,
                retrieved_chunks=retrieved_chunks,
                source_document_id=source_document_id,
                source_filename=source_filename,
            )
            if stream_meta is not None
            else ReviewMetadata(
                model=llm.model,
                latency_ms=total_ms,
                latency=latency,
                token_usage=None,
                retrieval_enabled=use_retrieval,
                retrieved_chunks=retrieved_chunks,
                source_document_id=source_document_id,
                source_filename=source_filename,
            )
        )
        input_tokens = metadata.token_usage.input_tokens if metadata.token_usage else 0
        output_tokens = metadata.token_usage.output_tokens if metadata.token_usage else 0
        record_review_success(
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=metadata.estimated_cost_usd or 0.0,
            retrieved_chunk_count=len(retrieved_chunks),
        )
        _log_completion(
            request_id=request_id,
            metadata=metadata,
            validation_path=validation_path,
            use_retrieval=use_retrieval,
        )

        yield StreamEvent(
            event="done",
            data={
                "review": review.model_dump(mode="json"),
                "request_id": request_id,
                "validation_path": validation_path,
                "metadata": metadata.model_dump(mode="json"),
            },
        )
    except LLMError as exc:
        record_review_failure()
        latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
        log_review_event(
            "review_failed",
            request_id,
            error_type=exc.error_type,
            retryable=exc.retryable,
            total_ms=latency_ms,
        )
        yield StreamEvent(
            event="error",
            data={
                **exc.to_dict(),
                "metadata": ReviewMetadata(
                    model=get_client().model,
                    latency_ms=latency_ms,
                    latency=LatencyBreakdown(total_ms=latency_ms),
                    token_usage=None,
                    retrieval_enabled=use_retrieval,
                    retrieved_chunks=retrieved_chunks,
                    source_document_id=source_document_id,
                    source_filename=source_filename,
                ).model_dump(mode="json"),
            },
        )
