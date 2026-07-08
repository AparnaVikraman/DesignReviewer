import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.models import (
    Category,
    DesignReview,
    Finding,
    Priority,
    ReviewMetadata,
    ReviewResponse,
    StreamEvent,
    TokenUsage,
)

client = TestClient(app)

SAMPLE_DOC = "Order service stores data in PostgreSQL and publishes events to Kafka."

SAMPLE_REVIEW = DesignReview(
    confidence=0.9,
    summary="Solid foundation with reliability gaps.",
    needs_human_review=True,
    findings=[
        Finding(
            category=Category.reliability,
            priority=Priority.high,
            summary="No retry strategy described.",
        )
    ],
)

SAMPLE_METADATA = ReviewMetadata(
    model="gpt-4.1-mini",
    latency_ms=42.0,
    token_usage=TokenUsage(input_tokens=100, output_tokens=200, total_tokens=300),
    retrieval_enabled=False,
    retrieved_chunks=[],
)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "reviews_total" in body


@patch("app.main.review_design")
def test_review_endpoint(mock_review_design):
    mock_review_design.return_value = ReviewResponse(
        review=SAMPLE_REVIEW,
        request_id="req-123",
        validation_path="direct",
        metadata=SAMPLE_METADATA,
    )

    response = client.post("/review", json={"design_doc": SAMPLE_DOC})

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-123"
    assert body["metadata"]["model"] == "gpt-4.1-mini"
    assert body["metadata"]["latency_ms"] == 42.0
    assert body["metadata"]["token_usage"]["total_tokens"] == 300
    assert body["review"]["confidence"] == 0.9
    mock_review_design.assert_called_once()
    kwargs = mock_review_design.call_args.kwargs
    assert kwargs["use_retrieval"] is True
    assert kwargs["source_document_id"] is None
    assert kwargs["source_filename"] is None


@patch("app.main.review_design")
def test_review_endpoint_without_retrieval(mock_review_design):
    mock_review_design.return_value = ReviewResponse(
        review=SAMPLE_REVIEW,
        request_id="req-456",
        validation_path="direct",
        metadata=SAMPLE_METADATA,
    )

    response = client.post(
        "/review",
        json={"design_doc": SAMPLE_DOC, "use_retrieval": False},
    )

    assert response.status_code == 200
    mock_review_design.assert_called_once()
    kwargs = mock_review_design.call_args.kwargs
    assert kwargs["use_retrieval"] is False


@patch("app.main.stream_review_design")
def test_review_stream_endpoint(mock_stream_review_design):
    mock_stream_review_design.return_value = iter(
        [
            StreamEvent(event="delta", data={"text": '{"confidence":'}),
            StreamEvent(event="delta", data={"text": "0.9}"}),
            StreamEvent(
                event="done",
                data={
                    "review": SAMPLE_REVIEW.model_dump(mode="json"),
                    "request_id": "req-stream-1",
                    "validation_path": "direct",
                    "metadata": SAMPLE_METADATA.model_dump(mode="json"),
                },
            ),
        ]
    )

    with client.stream("POST", "/review/stream", json={"design_doc": SAMPLE_DOC}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = []
        event_name = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append(
                    {
                        "event": event_name,
                        "data": json.loads(line.removeprefix("data: ")),
                    }
                )

    assert events[0]["event"] == "delta"
    assert events[0]["data"]["text"] == '{"confidence":'
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["request_id"] == "req-stream-1"
    assert events[-1]["data"]["metadata"]["latency_ms"] == 42.0
    assert events[-1]["data"]["metadata"]["model"] == "gpt-4.1-mini"
    assert events[-1]["data"]["review"]["confidence"] == 0.9
    mock_stream_review_design.assert_called_once()
    kwargs = mock_stream_review_design.call_args.kwargs
    assert kwargs["use_retrieval"] is True


def test_upload_documents_requires_files():
    response = client.post("/documents", files=[])
    assert response.status_code == 422


@patch("app.main.resolve_review_input")
@patch("app.main.review_design")
def test_review_by_document_id(mock_review_design, mock_resolve):
    from app.models import ReviewResponse
    from app.review_input import ResolvedReviewInput

    mock_resolve.return_value = ResolvedReviewInput(
        design_doc=SAMPLE_DOC,
        source_document_id="550e8400-e29b-41d4-a716-446655440000",
        source_filename="order_service_design.md",
    )
    mock_review_design.return_value = ReviewResponse(
        review=SAMPLE_REVIEW,
        request_id="req-doc",
        validation_path="direct",
        metadata=SAMPLE_METADATA,
    )

    response = client.post(
        "/review",
        json={"document_id": "550e8400-e29b-41d4-a716-446655440000"},
    )

    assert response.status_code == 200
    mock_review_design.assert_called_once()
    kwargs = mock_review_design.call_args.kwargs
    assert kwargs["source_document_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert kwargs["source_filename"] == "order_service_design.md"
    assert "request_id" in kwargs


def test_review_requires_single_input_source():
    response = client.post("/review", json={})
    assert response.status_code == 422

    response = client.post(
        "/review",
        json={"design_doc": "test", "filename": "order.md"},
    )
    assert response.status_code == 422
