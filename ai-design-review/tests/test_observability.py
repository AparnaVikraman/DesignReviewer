from app.errors import ReviewError
from app.metrics import get_metrics, record_review_success, reset_metrics
from app.models import ErrorResponse
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_metrics_endpoint():
    reset_metrics()
    record_review_success(
        embedding_ms=320.0,
        retrieval_ms=18.0,
        llm_ms=2800.0,
        total_ms=3200.0,
        input_tokens=1450,
        output_tokens=640,
        estimated_cost_usd=0.001,
        retrieved_chunk_count=3,
    )

    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["reviews_total"] == 1
    assert body["latency_ms"]["embedding_avg"] == 320.0
    assert body["latency_ms"]["llm_avg"] == 2800.0
    assert body["tokens"]["input_total"] == 1450


def test_health_includes_review_counts():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "reviews_total" in body
    assert "reviews_failed" in body


def test_empty_design_doc_returns_structured_error():
    response = client.post("/review", json={"design_doc": "   "})
    assert response.status_code == 400
    body = response.json()
    assert body["error_type"] == "empty_design_doc"
    assert body["retryable"] is False
    assert "request_id" in body


def test_review_error_model():
    exc = ReviewError(
        "LLM request timed out",
        error_type="timeout",
        request_id="req-1",
        retryable=True,
        status_code=504,
    )
    payload = exc.to_response()
    assert isinstance(payload, ErrorResponse)
    assert payload.error == "LLM request timed out"
    assert payload.retryable is True
