from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Category(str, Enum):
    reliability = "reliability"
    scalability = "scalability"
    security = "security"
    observability = "observability"
    api_design = "api_design"
    data_consistency = "data_consistency"
    operational = "operational"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Citation(BaseModel):
    source_file: str = Field(min_length=1)
    title: str = ""


class Finding(BaseModel):
    category: Category
    priority: Priority
    summary: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class DesignReview(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1)
    needs_human_review: bool
    findings: list[Finding] = Field(min_length=1)


class ReviewRequest(BaseModel):
    design_doc: str = Field(min_length=1)
    use_retrieval: bool = True


class RetrievedChunk(BaseModel):
    source_file: str
    title: str
    content: str
    score: float
    chunk_index: int


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ReviewMetadata(BaseModel):
    model: str
    latency_ms: float
    token_usage: TokenUsage | None = None
    retrieval_enabled: bool = False
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    review: DesignReview
    request_id: str
    validation_path: Literal["direct", "repair", "fallback"] | None = None
    metadata: ReviewMetadata


class CompareReviewResponse(BaseModel):
    design_doc: str
    without_retrieval: ReviewResponse
    with_retrieval: ReviewResponse


class StreamEvent(BaseModel):
    event: Literal["delta", "done", "error"]
    data: dict[str, Any]


def design_review_fallback() -> DesignReview:
    return DesignReview(
        confidence=0.0,
        summary=(
            "Automated review unavailable due to invalid model output. "
            "Manual architect review required."
        ),
        needs_human_review=True,
        findings=[
            Finding(
                category=Category.operational,
                priority=Priority.high,
                summary="LLM response failed Pydantic validation after repair attempts.",
            )
        ],
    )


def openai_json_schema(model: type[BaseModel]) -> dict[str, object]:
    schema = model.model_json_schema()
    return _ensure_strict_schema(schema)


def _ensure_strict_schema(schema: dict[str, object]) -> dict[str, object]:
    schema = dict(schema)
    schema.pop("title", None)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["required"] = list(properties.keys())
        schema["additionalProperties"] = False

    defs = schema.get("$defs")
    if isinstance(defs, dict):
        schema["$defs"] = {
            name: _ensure_strict_schema(defn) if isinstance(defn, dict) else defn
            for name, defn in defs.items()
        }

    return schema
