# Rate limits and quotas

## Request rates
The Free plan allows 60 requests per minute, Pro allows 600 requests per minute and Enterprise allows 6000 requests per minute. Short bursts of up to twice the limit are tolerated for 10 seconds.

## Hitting the limit
When you exceed the limit the API returns HTTP 429 and a `Retry-After` header that says how many seconds to wait. Clients should back off exponentially rather than retrying immediately.

## Payloads and pagination
The maximum request body is 5 MB. List endpoints return 50 items by default and accept up to 200 items per page. The batch events endpoint accepts at most 100 events per call.
