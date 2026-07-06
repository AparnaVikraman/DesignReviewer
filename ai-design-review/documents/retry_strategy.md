# Retry Strategy

## Principles

- Retry only **transient** failures (network blips, 503, connection timeouts).
- Do **not** retry non-idempotent operations without an idempotency key.
- Use **exponential backoff with jitter** to avoid thundering herds.
- Cap total retry attempts (typically 3–5) and enforce a **deadline** per request chain.

## HTTP / API Clients

- Retry on: 408, 429 (with Retry-After), 500, 502, 503, 504.
- Do not retry on: 400, 401, 403, 404, 422 (client errors).
- Set connect and read timeouts explicitly — never rely on infinite defaults.

## Database Writes

- Wrap multi-step writes in a transaction.
- On transient deadlock or serialization failure, retry with backoff.
- For order creation, combine idempotency keys with at-most-once semantics at the API layer.

## Message Publishing (Kafka)

- Use producer retries with `acks=all` for critical events like OrderCreated.
- If publish fails after DB commit, use the **outbox pattern**: write event to an outbox table in the same transaction, then a relay process publishes to Kafka.
- Never silently drop failed publishes — route to a dead-letter topic after max retries.

## Recommended Defaults

| Setting | Value |
|---------|-------|
| Max retries | 3 |
| Base backoff | 100ms |
| Max backoff | 30s |
| Jitter | ±25% |
