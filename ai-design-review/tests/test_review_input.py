from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.chunk_service import ChunkService
from app.document_service import DocumentService
from app.main import app
from app.vector_store import InMemoryVectorStore

client = TestClient(app)


def test_review_by_filename_uses_uploaded_file(tmp_path: Path):
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    path = uploads_dir / "order_service_design.md"
    path.write_text(
        "# Order Service\n\nUses Kafka with no retry strategy.",
        encoding="utf-8",
    )

    store = InMemoryVectorStore()
    store.initialize()

    class FakeEmbeddingService:
        def generate_embedding(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    service = DocumentService(
        uploads_dir=uploads_dir,
        chunk_service=ChunkService(chunk_words=5),
        vector_store=store,
        embedding_service=FakeEmbeddingService(),  # type: ignore[arg-type]
    )
    record = service.process_file(path)

    with patch("app.review_input.DocumentService", return_value=service), patch(
        "app.review_service.get_client"
    ) as mock_client:
        from app.models import DesignReview, Category, Priority, Finding

        mock_client.return_value.complete.return_value = type(
            "Result",
            (),
            {
                "validated": DesignReview(
                    confidence=0.9,
                    summary="Review complete.",
                    needs_human_review=False,
                    findings=[
                        Finding(
                            category=Category.reliability,
                            priority=Priority.high,
                            summary="Missing retry strategy.",
                        )
                    ],
                ),
                "model": "gpt-test",
                "usage": None,
                "request_id": "req-1",
                "validation_path": "direct",
                "latency_ms": 1.0,
            },
        )()

        response = client.post(
            "/review",
            json={"filename": "order_service_design.md", "use_retrieval": False},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["source_filename"] == "order_service_design.md"
    assert body["metadata"]["source_document_id"] == record.document_id


def test_review_by_document_id_returns_404_when_missing():
    response = client.post(
        "/review",
        json={"document_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert response.status_code == 404
