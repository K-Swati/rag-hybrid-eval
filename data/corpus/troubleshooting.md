# Troubleshooting

## Agent not reporting
First check that outbound traffic on port 8126 is allowed. The agent log is written to `/var/log/nimbus-agent.log`. Set `NIMBUS_LOG_LEVEL=debug` for more detail and then run `nimbus-agent restart`.

## Common error codes
Error `E1001` means the host clock is skewed by more than 5 minutes; fix it with NTP. Error `E2040` means a duplicate metric name was registered with a different type.

## Cardinality
Each metric may have at most 10,000 unique tag combinations. Beyond that new series are dropped.
