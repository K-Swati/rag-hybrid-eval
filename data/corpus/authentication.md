# Authentication

## API keys
Every request must include an API key in the `Authorization: Bearer` header. Live keys start with `nk_live_` and test keys start with `nk_test_`. We recommend rotating keys every 90 days. Keys can be revoked instantly from the Security page.

## OAuth 2.0
Server-to-server integrations should use the OAuth 2.0 client credentials flow. Access tokens expire after 3600 seconds and must be refreshed by requesting a new token. Available scopes are `read:metrics` and `write:alerts`.

## Error responses
A missing or invalid key returns HTTP 401. A valid key without the required scope returns HTTP 403 with the code `insufficient_scope`.
