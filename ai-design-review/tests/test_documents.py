from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.chunk_service import ChunkService
from app.document_service import DocumentService
from app.main import app
from app.parser import DocumentParser
from app.vector_store import InMemoryVectorStore

client = TestClient(app)


def test_document_parser_reads_markdown(tmp_path: Path):
    path = tmp_path / "inventory.md"
    path.write_text("# Inventory\n\nTrack stock levels.", encoding="utf-8")

    parsed = DocumentParser().parse(path)

    assert parsed.filename == "inventory.md"
    assert parsed.format == "markdown"
    assert len(parsed.pages) == 1
    assert parsed.pages[0].title == "Inventory"
    assert "Track stock levels" in parsed.pages[0].text


def test_document_parser_splits_markdown_sections(tmp_path: Path):
    path = tmp_path / "order_service_design.md"
    path.write_text(
        "# Order Service\n\nOverview text.\n\n## API\n\nPOST /orders\n\n## Kafka\n\nPublishes events.",
        encoding="utf-8",
    )

    parsed = DocumentParser().parse(path)

    assert parsed.format == "markdown"
    assert len(parsed.pages) == 3
    assert parsed.pages[0].title == "Order Service"
    assert parsed.pages[1].title == "API"
    assert parsed.pages[2].title == "Kafka"
    assert "POST /orders" in parsed.pages[1].text


def test_normalize_upload_filename_for_markdown_content_type():
    from app.parser import normalize_upload_filename

    assert normalize_upload_filename("order_service_design", "text/markdown") == (
        "order_service_design.md"
    )


def test_document_parser_reads_text(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("Plain text design notes.", encoding="utf-8")

    parsed = DocumentParser().parse(path)

    assert parsed.format == "text"
    assert parsed.pages[0].text == "Plain text design notes."


def test_document_service_processes_markdown(tmp_path: Path):
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    path = uploads_dir / "payment_design.md"
    path.write_text(
        "# Payment Design\n\nProcess payments with Kafka and PostgreSQL.",
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

    assert record.filename == "payment_design.md"
    assert record.format == "markdown"
    assert record.chunk_count >= 1
    assert len(store.list_documents()) == 1
    assert store.similarity_search([1.0, 0.0, 0.0], top_k=1)[0].page_number == 1


@patch("app.main.DocumentService")
def test_upload_documents_endpoint(mock_service_cls):
    from app.models import DocumentRecord

    mock_service = mock_service_cls.return_value
    mock_service.upload_and_process_many.return_value = [
        DocumentRecord(
            document_id="doc-1",
            filename="inventory.md",
            format="markdown",
            page_count=1,
            chunk_count=2,
        )
    ]

    response = client.post(
        "/documents",
        files=[("files", ("inventory.md", BytesIO(b"# Inventory"), "text/markdown"))],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["documents"][0]["document_id"] == "doc-1"
    assert body["documents"][0]["filename"] == "inventory.md"
    assert body["documents"][0]["chunk_count"] == 2


def test_upload_documents_requires_files():
    response = client.post("/documents", files=[])
    assert response.status_code == 422
