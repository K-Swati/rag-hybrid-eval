# Webhooks

## Signatures
Every delivery is signed with HMAC-SHA256 and the signature is sent in the `X-Nimbus-Signature` header. Reject any request whose signature does not match, and reject timestamps older than 5 minutes to prevent replay attacks.

## Delivery and retries
Your endpoint must answer with a 2xx status within 10 seconds. Failed deliveries are retried up to 5 times with exponential backoff starting at 30 seconds. Delivery logs are kept for 7 days.

## Event types
Supported events are `alert.triggered`, `alert.resolved` and `deployment.created`.
