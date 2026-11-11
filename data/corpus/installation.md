# Installing the agent

## Requirements
The agent needs Linux kernel 4.15 or newer and at least 256 MB of RAM. The configuration file lives at `/etc/nimbus/agent.yaml`.

## Packages and containers
Install with apt or yum, with Homebrew on macOS, or run the `nimbus/agent` Docker image. On Kubernetes use the Helm chart with `helm install nimbus-agent`.

## Authentication
Provide the API key through the `NIMBUS_API_KEY` environment variable instead of writing it into the config file.
