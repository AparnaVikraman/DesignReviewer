# API Design Guidelines

## Idempotency

- All **create** endpoints that can be retried must accept an `Idempotency-Key` header.
- Store idempotency keys with TTL (24–72 hours) and return the same response for duplicate keys.
- Without idempotency, duplicate POSTs can create duplicate orders — a critical data consistency bug.

## Validation

- Validate input at the API boundary with clear 400 responses.
- Return structured error bodies: `{ "error": "...", "field": "...", "code": "..." }`.
- Never leak internal stack traces to clients.

## Timeouts

- Define explicit timeouts at every hop: client → API gateway → service → database → broker.
- Document SLA targets (e.g., p99 < 500ms for order creation).
- Propagate deadline context where possible.

## Versioning

- Version public APIs (`/v1/orders`) from day one.
- Breaking changes require a new version, not silent schema changes.

## Error Responses

| Status | Use when |
|--------|----------|
| 400 | Invalid input |
| 409 | Conflict (duplicate idempotency key with different body) |
| 422 | Semantic validation failure |
| 503 | Dependency unavailable, safe to retry |

## Order Creation Checklist

- [ ] Idempotency-Key support
- [ ] Request validation schema
- [ ] Explicit timeout policy documented
- [ ] Structured error responses
- [ ] Rate limiting on public endpoints
