# Migrating from v1 to v2

## Timeline
API v1 is deprecated and will be shut down on March 31. All new features are only available in v2.

## What changed
List endpoints now use cursor pagination instead of page numbers. All timestamps are ISO 8601 in UTC. The field `ts` was renamed to `timestamp`, and the endpoint `/v1/metrics` is now `/v2/series`.

## Tooling
Run `nimbus migrate` to scan your code for v1 calls. While migrating you can send the `X-Nimbus-Compat` header to keep v1 response shapes.
