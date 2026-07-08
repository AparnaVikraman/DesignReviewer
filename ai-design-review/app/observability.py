import logging
from typing import Any

logger = logging.getLogger("app.review")


def log_review_event(event: str, request_id: str, **fields: Any) -> None:
    parts = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("event=%s request_id=%s %s", event, request_id, parts)


def log_retrieved_documents(
    request_id: str, chunks: list[Any], *, retrieval_enabled: bool
) -> None:
    if not retrieval_enabled:
        log_review_event("retrieval_skipped", request_id)
        return
    if not chunks:
        log_review_event("retrieval_empty", request_id, warning="no_guidance_chunks")
        return
    sources = ", ".join(getattr(chunk, "source_file", str(chunk)) for chunk in chunks)
    log_review_event(
        "retrieval_complete",
        request_id,
        chunk_count=len(chunks),
        sources=sources,
    )
