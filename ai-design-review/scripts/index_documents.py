#!/usr/bin/env python3
"""Index knowledge-base documents into the vector store."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from app.embedding_service import EmbeddingService

load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    service = EmbeddingService()
    count = service.index_documents()
    print(f"Indexed {count} chunks from documents/")


if __name__ == "__main__":
    main()
