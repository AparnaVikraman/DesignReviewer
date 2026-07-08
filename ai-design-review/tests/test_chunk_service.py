from app.chunk_service import ChunkService
from app.parser import ParsedDocument, ParsedPage


def test_chunk_service_splits_long_text():
    service = ChunkService(chunk_words=10)
    text = " ".join(f"word{i}" for i in range(25))
    chunks = service.chunk_text(text, page_number=2, start_chunk_number=0)

    assert len(chunks) == 3
    assert chunks[0].page_number == 2
    assert chunks[0].chunk_number == 0
    assert chunks[1].chunk_number == 1
    assert chunks[2].chunk_number == 2
    assert "word0" in chunks[0].content
    assert "word20" in chunks[2].content


def test_chunk_service_handles_multi_page_document():
    service = ChunkService(chunk_words=5)
    document = ParsedDocument(
        filename="inventory.md",
        format="markdown",
        pages=[
            ParsedPage(page_number=1, text="alpha beta gamma delta epsilon zeta"),
            ParsedPage(page_number=2, text="one two three four five six"),
        ],
    )

    chunks = service.chunk_document(document)

    assert len(chunks) == 4
    assert chunks[0].page_number == 1
    assert chunks[2].page_number == 2
    assert [chunk.chunk_number for chunk in chunks] == [0, 1, 2, 3]
