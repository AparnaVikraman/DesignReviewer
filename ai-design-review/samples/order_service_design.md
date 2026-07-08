# Order Service — Design Draft

**Author:** Your Name  
**Status:** Draft  
**Last updated:** 2026-03-10

## Overview

The Order Service handles order creation for our e-commerce platform. Clients call a REST API to create orders. Orders are persisted in PostgreSQL and an `OrderCreated` event is published to Kafka for downstream services (inventory, payment, fulfillment).

## Goals

- Accept order creation requests via HTTP
- Persist orders reliably
- Notify downstream systems asynchronously via Kafka

## API

### `POST /orders`

Creates a new order.

**Request body (example):**

```json
{
  "customer_id": "cust_123",
  "items": [
    { "sku": "SKU-001", "quantity": 2, "price_cents": 1999 }
  ],
  "shipping_address": {
    "line1": "123 Main St",
    "city": "San Francisco",
    "postal_code": "94105",
    "country": "US"
  }
}
```

**Response:** `201 Created` with order ID and status `PENDING`.

**Notes:**
- Authentication: API key in header (details TBD)
- No idempotency key required today
- Default HTTP client timeouts used by callers

## Data model

- **PostgreSQL** is the system of record for orders
- Table: `orders` (id, customer_id, status, total_cents, created_at)
- Table: `order_items` (order_id, sku, quantity, price_cents)

Order and items are written in a single database transaction.

## Event flow

1. Validate request
2. Insert order + items in PostgreSQL
3. Publish `OrderCreated` event to Kafka topic `orders.events`
4. Return response to client

Downstream consumers:
- **Inventory Service** — reserve stock
- **Payment Service** — charge customer
- **Fulfillment Service** — schedule shipment

## Kafka

- Topic: `orders.events`
- Event type: `OrderCreated`
- Payload includes order ID, customer ID, line items, timestamp
- Producer: fire-and-forget after DB commit
- No dead-letter queue configured yet
- No retry policy documented for producer or consumers

## Reliability & failure handling

- If DB write fails → return `500` to client
- If Kafka publish fails after DB commit → **not handled** (event may be lost)
- No outbox pattern
- No compensating transactions

## Security

- API key auth (rotation process not defined)
- Kafka: plaintext within VPC (TLS not specified)
- PII in events: customer ID and shipping address included

## Observability

- Application logs to stdout
- No metrics, dashboards, or alerts defined
- No consumer lag monitoring
- No tracing between API → DB → Kafka

## Scalability

- Single PostgreSQL instance
- Kafka cluster shared with other services
- No autoscaling strategy documented
- Peak load estimate: 500 orders/minute (Black Friday)

## Open questions

1. Should we require idempotency keys on `POST /orders`?
2. How do we handle duplicate events downstream?
3. What is the retry/DLQ strategy for Kafka?
4. What SLOs do we need for order creation latency?

## Out of scope (v1)

- Order cancellation
- Order updates
- Refunds

---

*Upload then review by `document_id` or `filename` — no need to paste text manually.*
