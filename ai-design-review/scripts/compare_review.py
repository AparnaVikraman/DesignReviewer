#!/usr/bin/env python3
"""Compare design review output with and without RAG retrieval."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from app.review_service import compare_reviews

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DESIGN_DOC = """
Order Service Design

The service receives order creation requests through a REST API.
It stores orders in PostgreSQL.
After saving the order, it publishes an OrderCreated event to Kafka.
Other services consume the event for inventory, payment, and fulfillment.

Current design:
- No retry strategy
- No dead-letter queue
- No idempotency key
- No explicit timeout policy
- No monitoring or alerting described
- PostgreSQL is the main source of truth
"""


def main() -> None:
    design_doc = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DESIGN_DOC.strip()
    result = compare_reviews(design_doc)

    print("\n=== WITHOUT RETRIEVAL ===")
    print(json.dumps(result.without_retrieval.model_dump(mode="json"), indent=2))

    print("\n=== WITH RETRIEVAL ===")
    print(f"Retrieved chunks: {len(result.with_retrieval.metadata.retrieved_chunks)}")
    for chunk in result.with_retrieval.metadata.retrieved_chunks:
        print(f"  - {chunk.source_file} ({chunk.title}) score={chunk.score}")
    review = result.with_retrieval.review
    for finding in review.findings:
        if finding.citations:
            cites = ", ".join(c.source_file for c in finding.citations)
            print(f"  [{finding.category}] cites: {cites}")
    print(json.dumps(result.with_retrieval.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
