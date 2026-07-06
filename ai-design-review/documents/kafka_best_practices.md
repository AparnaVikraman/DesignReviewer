# Kafka Best Practices

## Event Design

- Events must be **immutable facts** (OrderCreated, not OrderUpdatedInPlace).
- Include: event ID, timestamp, schema version, and correlation ID.
- Use Avro or JSON Schema with a schema registry for evolution.

## Reliability

- Producer: `acks=all`, enable idempotent producer, set `retries` and `delivery.timeout.ms`.
- Consumer: commit offsets **after** successful processing (at-least-once).
- Failed messages after max retries go to a **Dead Letter Queue (DLQ)** topic — never discard silently.

## Ordering

- Use partition keys (e.g., `order_id`) to preserve order per entity.
- Document ordering guarantees per topic.

## Outbox Pattern

For services that write to DB then publish:

1. Insert business record + outbox row in **one transaction**.
2. Separate relay reads outbox and publishes to Kafka.
3. Mark outbox row as published on success.

This avoids the dual-write problem (DB committed, Kafka publish failed).

## Monitoring

- Track consumer lag per group.
- Alert when lag exceeds threshold (e.g., > 10,000 messages or > 5 minutes).
- Monitor DLQ depth — sustained growth indicates a downstream failure.

## OrderCreated Event Checklist

- [ ] DLQ configured for poison messages
- [ ] Retry policy on producer and consumer
- [ ] Outbox or transactional publish pattern
- [ ] Schema versioning strategy
- [ ] Consumer lag alerting
