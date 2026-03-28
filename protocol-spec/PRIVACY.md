# WMCP Privacy Policy

## Telemetry

WMCP includes an optional, opt-in anonymous usage telemetry system.

### What is collected (when enabled)

- Protocol version
- Number of agents
- Domain names used
- Average latency
- Event types (e.g., "protocol_trained", "onboard_complete")
- A random installation ID (not linked to your identity)

### What is NOT collected

- Model weights or features
- Message content or tokens
- File paths or system information
- IP addresses or location data
- Personal information of any kind

### How to control telemetry

Telemetry is **disabled by default**. To enable or disable:

```bash
wmcp telemetry --enable    # Opt in
wmcp telemetry --disable   # Opt out
```

Or set the environment variable:
```bash
export WMCP_TELEMETRY=false  # Disable
export WMCP_TELEMETRY=true   # Enable
```

### Data storage

In v0.1, all telemetry data is stored **locally only** at `~/.wmcp/telemetry.jsonl`. No data is sent over the network. Future versions may add optional remote reporting to a configurable endpoint.

### Questions

Contact: t.kaszynski@proton.me
