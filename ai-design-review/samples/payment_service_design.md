# Payment Service — Design Draft

**Author:** Your Name  
**Status:** Draft

## Overview

The Payment Service processes payments when an `OrderCreated` event is received from Kafka. It charges the customer's payment method and emits `PaymentCompleted` or `PaymentFailed`.

## Flow

1. Consume `OrderCreated` from Kafka
2. Call external payment gateway (Stripe)
3. Update payment status in PostgreSQL
4. Publish result event to Kafka

## Current design gaps (intentional — edit as you test)

- No retry strategy for payment gateway calls
- No idempotency key when charging (risk of double charge on retry)
- Consumer offset committed before payment completes
- No dead-letter queue for poison messages
- No circuit breaker on gateway failures
- No metrics on payment success rate or latency
- No alert on payment failure spike

## Data

- PostgreSQL table: `payments` (order_id, status, amount_cents, gateway_ref)
- Events: `PaymentCompleted`, `PaymentFailed` on topic `payments.events`

## Security

- Gateway API keys stored in environment variables
- Card data never stored (tokenized by Stripe)

---

*Upload via: `curl -X POST http://localhost:8000/documents -F "files=@samples/payment_service_design.md"`*
