import logging
import time
import uuid
from collections.abc import Iterator

from app.llm_client import LLMClient, LLMError, StreamMetadata, usage_to_token_counts
from app.models import (
    CompareReviewResponse,
    DesignReview,
    RetrievedChunk,
    ReviewMetadata,
    ReviewResponse,
    StreamEvent,
    TokenUsage,
    design_review_fallback,
    openai_json_schema,
)
from app.prompt_builder import PromptBuilder
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
        _client = LLMClient()
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


def _build_metadata(
    *,
    model: str,
    latency_ms: float,
    usage: object | None,
    retrieval_enabled: bool,
    retrieved_chunks: list[RetrievedChunk],
) -> ReviewMetadata:
    input_tokens, output_tokens, total_tokens = usage_to_token_counts(usage)
    token_usage = None
    if total_tokens > 0:
        token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    return ReviewMetadata(
        model=model,
        latency_ms=latency_ms,
        token_usage=token_usage,
        retrieval_enabled=retrieval_enabled,
        retrieved_chunks=retrieved_chunks,
    )


def _metadata_from_stream(
    stream_meta: StreamMetadata,
    latency_ms: float,
    *,
    retrieval_enabled: bool,
    retrieved_chunks: list[RetrievedChunk],
) -> ReviewMetadata:
    token_usage = None
    if stream_meta.total_tokens > 0:
        token_usage = TokenUsage(
            input_tokens=stream_meta.input_tokens,
            output_tokens=stream_meta.output_tokens,
            total_tokens=stream_meta.total_tokens,
        )
    return ReviewMetadata(
        model=stream_meta.model,
        latency_ms=latency_ms,
        token_usage=token_usage,
        retrieval_enabled=retrieval_enabled,
        retrieved_chunks=retrieved_chunks,
    )


def _build_prompt(design_doc: str, use_retrieval: bool) -> tuple[str, list[RetrievedChunk]]:
    retrieved_chunks: list[RetrievedChunk] = []
    prompt_builder = get_prompt_builder()

    if use_retrieval:
        retrieved_chunks = get_retriever().retrieve(design_doc)
        prompt = prompt_builder.build(design_doc, retrieved_chunks)
    else:
        prompt = prompt_builder.build(design_doc)

    return prompt, retrieved_chunks


def review_design(design_doc: str, *, use_retrieval: bool = True) -> ReviewResponse:
    request_id = str(uuid.uuid4())
    prompt, retrieved_chunks = _build_prompt(design_doc, use_retrieval)
    llm = get_client()
    start_time = time.perf_counter()

    result = llm.complete(
        prompt,
        request_id=request_id,
        json_schema=DESIGN_REVIEW_SCHEMA,
        json_schema_name=JSON_SCHEMA_NAME,
        response_model=DesignReview,
        fallback=lambda _error: design_review_fallback(),
    )

    review = result.validated
    if not isinstance(review, DesignReview):
        raise LLMError(
            "Validated review missing from LLM response",
            error_type="bad_response",
            request_id=request_id,
            retryable=False,
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    metadata = _build_metadata(
        model=result.model,
        latency_ms=latency_ms,
        usage=result.usage,
        retrieval_enabled=use_retrieval,
        retrieved_chunks=retrieved_chunks,
    )
    logger.info(
        "Review completed request_id=%s latency_ms=%.2f llm_latency_ms=%.2f "
        "validation_path=%s model=%s retrieval=%s chunks=%d",
        result.request_id,
        metadata.latency_ms,
        result.latency_ms,
        result.validation_path,
        metadata.model,
        use_retrieval,
        len(retrieved_chunks),
    )

    return ReviewResponse(
        review=review,
        request_id=result.request_id,
        validation_path=result.validation_path,
        metadata=metadata,
    )


def compare_reviews(design_doc: str) -> CompareReviewResponse:
    logger.info("Running comparison review without retrieval")
    without_retrieval = review_design(design_doc, use_retrieval=False)
    logger.info("Running comparison review with retrieval")
    with_retrieval = review_design(design_doc, use_retrieval=True)
    return CompareReviewResponse(
        design_doc=design_doc,
        without_retrieval=without_retrieval,
        with_retrieval=with_retrieval,
    )


def stream_review_design(
    design_doc: str, *, use_retrieval: bool = True
) -> Iterator[StreamEvent]:
    request_id = str(uuid.uuid4())
    prompt, retrieved_chunks = _build_prompt(design_doc, use_retrieval)
    chunks: list[str] = []
    start_time = time.perf_counter()
    stream_meta: StreamMetadata | None = None

    try:
        llm = get_client()
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

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        metadata = (
            _metadata_from_stream(
                stream_meta,
                latency_ms,
                retrieval_enabled=use_retrieval,
                retrieved_chunks=retrieved_chunks,
            )
            if stream_meta is not None
            else ReviewMetadata(
                model=llm.model,
                latency_ms=latency_ms,
                token_usage=None,
                retrieval_enabled=use_retrieval,
                retrieved_chunks=retrieved_chunks,
            )
        )
        logger.info(
            "Stream review completed request_id=%s latency_ms=%.2f validation_path=%s "
            "model=%s retrieval=%s chunks=%d",
            request_id,
            metadata.latency_ms,
            validation_path,
            metadata.model,
            use_retrieval,
            len(retrieved_chunks),
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
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(
            "Stream review failed request_id=%s error_type=%s latency_ms=%.2f",
            request_id,
            exc.error_type,
            latency_ms,
        )
        yield StreamEvent(
            event="error",
            data={
                **exc.to_dict(),
                "metadata": ReviewMetadata(
                    model=get_client().model,
                    latency_ms=latency_ms,
                    token_usage=None,
                    retrieval_enabled=use_retrieval,
                    retrieved_chunks=retrieved_chunks,
                ).model_dump(mode="json"),
            },
        )
