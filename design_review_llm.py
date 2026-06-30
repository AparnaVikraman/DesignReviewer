# design_review_llm.py

import json
import logging
import sys

from llm_client import LLMClient, LLMError
from models import DesignReview, design_review_fallback, openai_json_schema

client = LLMClient()

DESIGN_REVIEW_SCHEMA = openai_json_schema(DesignReview)

DESIGN_DOC = """
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

PROMPT = f"""
You are a Principal Software Engineer reviewing a backend system design.

Review the design below. Return a JSON object with:
- confidence: your confidence in the assessment (0.0 to 1.0)
- summary: brief overall assessment including key strengths
- needs_human_review: true if a human architect should review before implementation
- findings: a list of concerns, one per relevant category. Each finding must include:
  - category: reliability, scalability, security, observability, api_design,
    data_consistency, or operational
  - priority: low, medium, high, or critical
  - summary: specific issue, risk, or gap for that category

Include a finding for every category where you identify an issue. Do not collapse
multiple categories into a single finding.

Design document:
{DESIGN_DOC}
"""

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        result = client.complete(
            PROMPT,
            request_id="design-review-order-service",
            json_schema=DESIGN_REVIEW_SCHEMA,
            json_schema_name="design_review",
            response_model=DesignReview,
            fallback=lambda _error: design_review_fallback(),
        )
    except LLMError as e:
        print("\n=== Error ===")
        print(f"Type: {e.error_type}")
        print(f"Message: {e.message}")
        print(f"Request ID: {e.request_id}")
        if e.api_request_id:
            print(f"API Request ID: {e.api_request_id}")
        print(f"Attempts: {e.attempts}")
        sys.exit(1)

    print("\n=== Project ===")
    print("AI Design Review Assistant")

    print("\n=== Model ===")
    print(result.model)

    print("\n=== Request ID ===")
    print(result.request_id)

    print("\n=== API Request ID ===")
    print(result.api_request_id)

    print("\n=== Validation ===")
    print(f"Path: {result.validation_path}")

    print("\n=== Latency ===")
    print(f"{result.latency_ms} ms")

    print("\n=== Token Usage ===")
    print(result.usage)

    print("\n=== Design Review ===")
    review = result.validated
    assert isinstance(review, DesignReview)
    print(f"Confidence: {review.confidence}")
    print(f"Needs human review: {review.needs_human_review}")
    print(f"\nOverall: {review.summary}")
    print(f"\nFindings ({len(review.findings)}):")
    for i, finding in enumerate(review.findings, 1):
        print(
            f"\n  {i}. [{finding.category.value}] "
            f"priority={finding.priority.value}\n"
            f"     {finding.summary}"
        )

    print("\n=== Raw JSON ===")
    print(json.dumps(review.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
