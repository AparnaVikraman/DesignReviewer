from app.models import RetrievedChunk
from app.prompts import PromptBuilder
from app.vector_store import InMemoryVectorStore, StoredChunk


def test_prompt_builder_without_retrieval():
    builder = PromptBuilder()
    prompt = builder.build_without_retrieval("Order service uses Kafka.")
    assert "Order service uses Kafka." in prompt
    assert "Relevant engineering guidance" not in prompt


def test_prompt_builder_with_retrieval():
    builder = PromptBuilder()
    chunks = [
        RetrievedChunk(
            source_file="retry_strategy.md",
            title="Retry Strategy",
            content="Use exponential backoff with jitter.",
            score=0.91,
            page_number=1,
            chunk_number=0,
        ),
        RetrievedChunk(
            source_file="kafka_best_practices.md",
            title="Kafka Best Practices",
            content="Configure a dead-letter queue for failed events.",
            score=0.88,
            page_number=1,
            chunk_number=1,
        ),
    ]
    prompt = builder.build_with_retrieval("Order service uses Kafka.", chunks)
    assert "source_file: retry_strategy.md" in prompt
    assert "page_number: 1" in prompt
    assert "chunk_number: 0" in prompt
    assert "exponential backoff" in prompt
    assert "dead-letter queue" in prompt
    assert "Order service uses Kafka." in prompt


def test_finding_citations_schema():
    from app.models import Citation, Finding, openai_json_schema, DesignReview

    finding = Finding(
        category="reliability",
        priority="high",
        summary="Missing retry strategy.",
        citations=[
            Citation(
                source_file="retry_strategy.md",
                title="Message Publishing (Kafka)",
                page_number=1,
                chunk_number=4,
            )
        ],
    )
    assert finding.citations[0].chunk_number == 4

    schema = openai_json_schema(DesignReview)
    finding_props = schema["$defs"]["Finding"]["properties"]
    assert "citations" in finding_props


def test_retriever_returns_top_chunks():
    from app.retriever import Retriever

    store = InMemoryVectorStore()
    store.initialize()
    store.upsert_chunks(
        [
            StoredChunk(
                id=1,
                document_id="doc-1",
                source_file="retry_strategy.md",
                title="Retry Strategy",
                page_number=1,
                chunk_number=0,
                content="Use retries with exponential backoff.",
                embedding=[1.0, 0.0, 0.0],
            ),
            StoredChunk(
                id=2,
                document_id="doc-2",
                source_file="kafka_best_practices.md",
                title="Kafka Best Practices",
                page_number=1,
                chunk_number=0,
                content="Use DLQ for poison messages.",
                embedding=[0.0, 1.0, 0.0],
            ),
        ]
    )

    class FakeEmbeddingService:
        def generate_embedding(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    retriever = Retriever(
        vector_store=store,
        embedding_service=FakeEmbeddingService(),  # type: ignore[arg-type]
        top_k=1,
    )
    results = retriever.retrieve("order retry policy")
    assert len(results.chunks) == 1
    assert results.chunks[0].source_file == "retry_strategy.md"
    assert results.embedding_ms >= 0
