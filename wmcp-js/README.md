# wmcp-client

JavaScript/TypeScript client for the WMCP protocol server.

## Install

```bash
npm install wmcp-client
```

## Usage

```javascript
const { WMCPClient } = require('wmcp-client');

const client = new WMCPClient('http://localhost:8000');

// Health check
const health = await client.health();
console.log(health);  // { status: 'healthy', protocol_loaded: true, ... }

// Run communication round
const result = await client.communicate();
console.log(result.tokens_a);  // [1, 2]
console.log(result.a_greater); // true
console.log(result.latency_ms); // 0.85

// Get Prometheus metrics
const metrics = await client.metrics();
```

## Prerequisites

Start the WMCP server:
```bash
pip install wmcp fastapi uvicorn
python -m wmcp.server
```

## API

| Method | Description |
|--------|-------------|
| `health()` | Server health status |
| `communicate()` | Run communication round |
| `validate()` | Run compliance check |
| `metrics()` | Prometheus metrics (text) |
| `dashboard()` | Health dashboard (HTML) |
