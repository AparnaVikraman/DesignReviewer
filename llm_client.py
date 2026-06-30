import json
import logging
import os
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import tiktoken
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# USD per 1M tokens
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}
DEFAULT_PRICING = {"input": 1.00, "output": 3.00}

RETRYABLE_STATUS_CODES = {408, 409, 500, 502, 503, 504}


class LLMError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        request_id: str,
        api_request_id: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        attempts: int = 1,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.request_id = request_id
        self.api_request_id = api_request_id
        self.status_code = status_code
        self.retryable = retryable
        self.attempts = attempts
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "error_type": self.error_type,
            "request_id": self.request_id,
            "api_request_id": self.api_request_id,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "attempts": self.attempts,
        }

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    request_id: str
    api_request_id: str | None
    response_id: str
    usage: Any
    latency_ms: float
    input_tokens_estimated: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    parsed: dict[str, Any] | None = None
    validated: BaseModel | None = None
    validation_path: Literal["direct", "repair", "fallback"] | None = None


def _format_validation_errors(error: ValidationError) -> str:
    return "\n".join(
        f"- {'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
        for err in error.errors()
    )


def _build_text_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": True,
        }
    }


def _parse_json_response(text: str, *, request_id: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Model returned invalid JSON (request_id={request_id})",
            error_type="bad_response",
            request_id=request_id,
            retryable=False,
            cause=exc,
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMError(
            f"Model returned JSON that is not an object (request_id={request_id})",
            error_type="bad_response",
            request_id=request_id,
            retryable=False,
        )
    return parsed


def count_tokens(text: str, model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return round(
        (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000,
        6,
    )


def _parse_retry_after(headers: Any) -> float | None:
    if headers is None:
        return None
    retry_after = headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None


def _is_quota_error(exc: RateLimitError) -> bool:
    return "quota" in exc.message.lower()


def _map_error(
    exc: Exception,
    *,
    request_id: str,
    attempts: int,
    timeout: float,
) -> LLMError:
    if isinstance(exc, APITimeoutError):
        return LLMError(
            f"Request timed out after {timeout}s (request_id={request_id}, attempts={attempts})",
            error_type="timeout",
            request_id=request_id,
            retryable=True,
            attempts=attempts,
            cause=exc,
        )

    if isinstance(exc, APIConnectionError) and not isinstance(exc, APITimeoutError):
        return LLMError(
            f"Connection error (request_id={request_id}, attempts={attempts})",
            error_type="connection",
            request_id=request_id,
            retryable=True,
            attempts=attempts,
            cause=exc,
        )

    if isinstance(exc, RateLimitError):
        if _is_quota_error(exc):
            return LLMError(
                "OpenAI quota exceeded. Check billing and API key.",
                error_type="quota",
                request_id=request_id,
                api_request_id=exc.request_id,
                status_code=exc.status_code,
                retryable=False,
                attempts=attempts,
                cause=exc,
            )
        return LLMError(
            f"Rate limit exceeded (request_id={request_id}, attempts={attempts})",
            error_type="rate_limit",
            request_id=request_id,
            api_request_id=exc.request_id,
            status_code=exc.status_code,
            retryable=True,
            attempts=attempts,
            cause=exc,
        )

    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return LLMError(
            exc.message,
            error_type="auth",
            request_id=request_id,
            api_request_id=getattr(exc, "request_id", None),
            status_code=getattr(exc, "status_code", None),
            retryable=False,
            attempts=attempts,
            cause=exc,
        )

    if isinstance(exc, (BadRequestError, UnprocessableEntityError)):
        return LLMError(
            exc.message,
            error_type="bad_request",
            request_id=request_id,
            api_request_id=exc.request_id,
            status_code=exc.status_code,
            retryable=False,
            attempts=attempts,
            cause=exc,
        )

    if isinstance(exc, InternalServerError) or (
        isinstance(exc, APIStatusError) and exc.status_code in RETRYABLE_STATUS_CODES
    ):
        return LLMError(
            exc.message if isinstance(exc, APIStatusError) else "Internal server error",
            error_type="server_error",
            request_id=request_id,
            api_request_id=getattr(exc, "request_id", None),
            status_code=getattr(exc, "status_code", None),
            retryable=True,
            attempts=attempts,
            cause=exc,
        )

    if isinstance(exc, APIStatusError):
        retryable = exc.status_code in RETRYABLE_STATUS_CODES
        return LLMError(
            exc.message,
            error_type="unknown",
            request_id=request_id,
            api_request_id=exc.request_id,
            status_code=exc.status_code,
            retryable=retryable,
            attempts=attempts,
            cause=exc,
        )

    return LLMError(
        str(exc),
        error_type="unknown",
        request_id=request_id,
        retryable=False,
        attempts=attempts,
        cause=exc,
    )


def _compute_backoff(
    attempt: int,
    *,
    backoff_factor: float,
    backoff_max: float,
    retry_after: float | None = None,
) -> float:
    if retry_after is not None and retry_after > 0:
        return min(backoff_max, retry_after)
    delay = min(backoff_max, backoff_factor * (2**attempt))
    return delay * random.uniform(0.5, 1.0)


class LLMClient:
    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.2,
        timeout: float = 60.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        backoff_max: float = 30.0,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.backoff_max = backoff_max
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout,
            max_retries=0,
        )

    def _create_response(
        self,
        *,
        resolved_model: str,
        input: str,
        temperature: float,
        resolved_max_tokens: int,
        resolved_timeout: float,
        request_id: str,
        json_schema: dict[str, Any] | None = None,
        json_schema_name: str = "structured_response",
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "input": input,
            "temperature": temperature,
            "max_output_tokens": resolved_max_tokens,
            "timeout": resolved_timeout,
            "metadata": {"request_id": request_id},
        }
        if json_schema is not None:
            kwargs["text"] = _build_text_format(json_schema_name, json_schema)
        return self._client.responses.with_raw_response.create(**kwargs)

    def _repair_response(
        self,
        *,
        invalid_text: str,
        validation_error: ValidationError,
        resolved_model: str,
        resolved_temperature: float,
        resolved_max_tokens: int,
        resolved_timeout: float,
        request_id: str,
        json_schema: dict[str, Any],
        json_schema_name: str,
    ) -> str:
        repair_prompt = (
            "The previous JSON response failed validation. "
            "Return corrected JSON only, matching the required schema.\n\n"
            f"Validation errors:\n{_format_validation_errors(validation_error)}\n\n"
            f"Invalid response:\n{invalid_text}"
        )
        logger.warning(
            "LLM response repair starting request_id=%s errors=%d",
            request_id,
            validation_error.error_count(),
        )
        raw = self._create_response(
            resolved_model=resolved_model,
            input=repair_prompt,
            temperature=resolved_temperature,
            resolved_max_tokens=resolved_max_tokens,
            resolved_timeout=resolved_timeout,
            request_id=f"{request_id}-repair",
            json_schema=json_schema,
            json_schema_name=json_schema_name,
        )
        response = raw.parse()
        return response.output_text

    def _validate_with_repair_or_fallback(
        self,
        *,
        text: str,
        parsed: dict[str, Any],
        response_model: type[BaseModel],
        fallback: Callable[[ValidationError], BaseModel] | None,
        max_repair_attempts: int,
        resolved_model: str,
        resolved_temperature: float,
        resolved_max_tokens: int,
        resolved_timeout: float,
        request_id: str,
        json_schema: dict[str, Any],
        json_schema_name: str,
    ) -> tuple[BaseModel, dict[str, Any], str, str]:
        try:
            validated = response_model.model_validate(parsed)
            return validated, parsed, text, "direct"
        except ValidationError as initial_error:
            logger.warning(
                "LLM response validation failed request_id=%s errors=%d",
                request_id,
                initial_error.error_count(),
            )

        last_error = initial_error
        for repair_attempt in range(max_repair_attempts):
            try:
                repaired_text = self._repair_response(
                    invalid_text=text,
                    validation_error=last_error,
                    resolved_model=resolved_model,
                    resolved_temperature=resolved_temperature,
                    resolved_max_tokens=resolved_max_tokens,
                    resolved_timeout=resolved_timeout,
                    request_id=request_id,
                    json_schema=json_schema,
                    json_schema_name=json_schema_name,
                )
                repaired = _parse_json_response(repaired_text, request_id=request_id)
                validated = response_model.model_validate(repaired)
                logger.info(
                    "LLM response repair succeeded request_id=%s attempt=%d",
                    request_id,
                    repair_attempt + 1,
                )
                return validated, repaired, repaired_text, "repair"
            except (ValidationError, LLMError) as exc:
                if isinstance(exc, ValidationError):
                    last_error = exc
                    logger.warning(
                        "LLM response repair failed request_id=%s attempt=%d errors=%d",
                        request_id,
                        repair_attempt + 1,
                        exc.error_count(),
                    )
                else:
                    logger.warning(
                        "LLM response repair request failed request_id=%s attempt=%d error_type=%s",
                        request_id,
                        repair_attempt + 1,
                        exc.error_type,
                    )

        if fallback is not None:
            validated = fallback(last_error)
            logger.warning(
                "LLM response using fallback request_id=%s after %d repair attempt(s)",
                request_id,
                max_repair_attempts,
            )
            return validated, validated.model_dump(mode="json"), text, "fallback"

        raise LLMError(
            f"Response validation failed after {max_repair_attempts} repair attempt(s) "
            f"(request_id={request_id})",
            error_type="validation_error",
            request_id=request_id,
            retryable=False,
            cause=last_error,
        )

    def complete(
        self,
        input: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        request_id: str | None = None,
        json_schema: dict[str, Any] | None = None,
        json_schema_name: str = "structured_response",
        response_model: type[BaseModel] | None = None,
        fallback: Callable[[ValidationError], BaseModel] | None = None,
        max_repair_attempts: int = 1,
    ) -> LLMResponse:
        request_id = request_id or str(uuid.uuid4())
        resolved_model = model or self.model
        resolved_max_tokens = max_tokens or self.max_tokens
        resolved_timeout = timeout or self.timeout
        resolved_temperature = temperature if temperature is not None else self.temperature
        input_tokens_estimated = count_tokens(input, resolved_model)
        total_attempts = self.max_retries + 1

        logger.info(
            "LLM request starting request_id=%s model=%s input_tokens_estimated=%d max_output_tokens=%d",
            request_id,
            resolved_model,
            input_tokens_estimated,
            resolved_max_tokens,
        )

        start_time = time.perf_counter()
        raw = None
        last_error: LLMError | None = None

        for attempt in range(total_attempts):
            try:
                raw = self._create_response(
                    resolved_model=resolved_model,
                    input=input,
                    temperature=resolved_temperature,
                    resolved_max_tokens=resolved_max_tokens,
                    resolved_timeout=resolved_timeout,
                    request_id=request_id,
                    json_schema=json_schema,
                    json_schema_name=json_schema_name,
                )
                break
            except Exception as exc:
                llm_error = _map_error(
                    exc,
                    request_id=request_id,
                    attempts=attempt + 1,
                    timeout=resolved_timeout,
                )
                last_error = llm_error

                if not llm_error.retryable or attempt >= self.max_retries:
                    logger.error(
                        "LLM request failed request_id=%s error_type=%s attempts=%d message=%s",
                        request_id,
                        llm_error.error_type,
                        llm_error.attempts,
                        llm_error.message,
                    )
                    raise llm_error from exc

                retry_after = None
                if isinstance(exc, RateLimitError):
                    retry_after = _parse_retry_after(exc.response.headers)

                delay = _compute_backoff(
                    attempt,
                    backoff_factor=self.backoff_factor,
                    backoff_max=self.backoff_max,
                    retry_after=retry_after,
                )
                logger.warning(
                    "LLM request retrying request_id=%s attempt=%d/%d delay=%.1fs error_type=%s",
                    request_id,
                    attempt + 2,
                    total_attempts,
                    delay,
                    llm_error.error_type,
                )
                time.sleep(delay)

        if raw is None:
            assert last_error is not None
            raise last_error

        response = raw.parse()
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        estimated_cost_usd = estimate_cost_usd(resolved_model, input_tokens, output_tokens)
        output_text = response.output_text
        parsed: dict[str, Any] | None = None
        validated: BaseModel | None = None
        validation_path: Literal["direct", "repair", "fallback"] | None = None

        if json_schema is not None:
            parsed = _parse_json_response(output_text, request_id=request_id)

        if response_model is not None:
            if parsed is None:
                raise LLMError(
                    f"response_model requires json_schema (request_id={request_id})",
                    error_type="bad_request",
                    request_id=request_id,
                    retryable=False,
                )
            if json_schema is None:
                raise LLMError(
                    f"response_model requires json_schema (request_id={request_id})",
                    error_type="bad_request",
                    request_id=request_id,
                    retryable=False,
                )
            validated, parsed, output_text, validation_path = self._validate_with_repair_or_fallback(
                text=output_text,
                parsed=parsed,
                response_model=response_model,
                fallback=fallback,
                max_repair_attempts=max_repair_attempts,
                resolved_model=resolved_model,
                resolved_temperature=resolved_temperature,
                resolved_max_tokens=resolved_max_tokens,
                resolved_timeout=resolved_timeout,
                request_id=request_id,
                json_schema=json_schema,
                json_schema_name=json_schema_name,
            )

        logger.info(
            "LLM request completed request_id=%s model=%s "
            "input_tokens_estimated=%d input_tokens=%d output_tokens=%d total_tokens=%d "
            "estimated_cost_usd=%.6f latency_ms=%.2f",
            request_id,
            resolved_model,
            input_tokens_estimated,
            input_tokens,
            output_tokens,
            total_tokens,
            estimated_cost_usd,
            latency_ms,
        )

        if validation_path:
            logger.info(
                "LLM response validation path=%s request_id=%s",
                validation_path,
                request_id,
            )

        return LLMResponse(
            text=output_text,
            model=response.model,
            request_id=request_id,
            api_request_id=raw.headers.get("x-request-id"),
            response_id=response.id,
            usage=usage,
            latency_ms=latency_ms,
            input_tokens_estimated=input_tokens_estimated,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            parsed=parsed,
            validated=validated,
            validation_path=validation_path,
        )
