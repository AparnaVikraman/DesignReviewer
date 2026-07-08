from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from app.llm_client import LLMError
from app.models import ErrorResponse


class ReviewError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        request_id: str,
        retryable: bool = False,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.request_id = request_id
        self.retryable = retryable
        self.status_code = status_code

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            error=self.message,
            error_type=self.error_type,
            request_id=self.request_id,
            retryable=self.retryable,
        )


def _llm_status_code(exc: LLMError) -> int:
    if exc.error_type == "timeout":
        return 504
    if exc.error_type == "rate_limit":
        return 429
    if exc.error_type in {"auth", "permission"}:
        return 401
    if exc.error_type == "bad_request":
        return 400
    if exc.retryable:
        return 503
    return 502


async def review_error_handler(_request: Request, exc: ReviewError) -> JSONResponse:
    body = exc.to_response().model_dump(mode="json")
    return JSONResponse(status_code=exc.status_code, content=body)


async def llm_error_handler(_request: Request, exc: LLMError) -> JSONResponse:
    body = ErrorResponse(
        error=exc.message,
        error_type=exc.error_type,
        request_id=exc.request_id,
        retryable=exc.retryable,
    ).model_dump(mode="json")
    return JSONResponse(status_code=_llm_status_code(exc), content=body)


def error_payload_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    return ErrorResponse(
        error=str(data.get("message") or data.get("error", "Unknown error")),
        error_type=str(data.get("error_type", "unknown")),
        request_id=str(data.get("request_id", "unknown")),
        retryable=bool(data.get("retryable", False)),
    ).model_dump(mode="json")
