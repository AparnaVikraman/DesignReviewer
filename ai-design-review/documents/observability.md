# Observability

## Three Pillars

Every production service needs **logs**, **metrics**, and **traces**.

## Logging

- Use structured JSON logs with: `timestamp`, `level`, `service`, `request_id`, `trace_id`.
- Log business events: order created, event published, publish failed.
- Never log PII or secrets.

## Metrics

Minimum metrics for an order service:

| Metric | Type | Purpose |
|--------|------|---------|
| `http_requests_total` | Counter | Request volume by status |
| `http_request_duration_seconds` | Histogram | Latency SLIs |
| `orders_created_total` | Counter | Business throughput |
| `kafka_publish_failures_total` | Counter | Event pipeline health |
| `db_connection_pool_active` | Gauge | DB saturation |

## Tracing

- Propagate `trace_id` from API through DB and Kafka publish.
- Use OpenTelemetry for instrumentation.
- Sample 100% in staging, 1–10% in production depending on volume.

## Alerting

Define alerts before launch:

- Error rate > 1% for 5 minutes
- p99 latency > SLA threshold
- Kafka consumer lag above threshold
- DLQ message rate > 0 sustained

## Dashboards

- Golden signals dashboard: latency, traffic, errors, saturation.
- Business dashboard: orders/min, failure reasons, retry counts.

## Order Service Observability Checklist

- [ ] Structured logging with request IDs
- [ ] RED metrics (Rate, Errors, Duration)
- [ ] Distributed tracing across API → DB → Kafka
- [ ] Alerting runbook linked from each alert
- [ ] On-call escalation path documented
