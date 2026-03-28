"""
WMCP API Server (FastAPI)
==========================
REST API for protocol operations: validate, onboard, communicate, monitor.

Usage:
    pip install fastapi uvicorn
    python -m wmcp.server
    # Or: uvicorn wmcp.server:app --port 8000
"""

import time
import json
import io
import numpy as np

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import PlainTextResponse, HTMLResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

import torch
from wmcp.protocol import Protocol
from wmcp.analytics import ProtocolMetrics, export_prometheus, generate_health_report


def create_app():
    """Create the FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="WMCP API",
        description="World Model Communication Protocol — REST API",
        version="0.1.0",
    )

    # In-memory protocol state
    state = {
        "protocol": None,
        "metrics_history": [],
        "start_time": time.time(),
        "request_count": 0,
    }

    @app.get("/")
    async def root():
        return {
            "service": "WMCP API",
            "version": "0.1.0",
            "protocol_loaded": state["protocol"] is not None,
            "uptime_s": time.time() - state["start_time"],
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "protocol_loaded": state["protocol"] is not None,
            "requests_served": state["request_count"],
            "uptime_s": time.time() - state["start_time"],
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        m = ProtocolMetrics(
            timestamp=time.time(),
            n_agents=state["protocol"].n_agents if state["protocol"] else 0,
            total_messages=state["request_count"],
            accuracy=0.0,
            posdis=0.0,
            topsim=0.0,
            bosdis=0.0,
            latency_mean_ms=0.0,
            latency_p95_ms=0.0,
            entropy_mean=0.0,
        )
        return export_prometheus(m)

    @app.post("/validate")
    async def validate():
        """Run compliance validation on loaded protocol."""
        if state["protocol"] is None:
            raise HTTPException(400, "No protocol loaded")
        state["request_count"] += 1
        return {"status": "compliance check requires agent_views — use CLI instead",
                "protocol_info": {
                    "n_agents": state["protocol"].n_agents,
                    "vocab_size": state["protocol"].vocab_size,
                    "msg_dim": state["protocol"].msg_dim,
                }}

    @app.post("/communicate")
    async def communicate():
        """Run a communication round on synthetic data."""
        if state["protocol"] is None:
            # Create a demo protocol
            state["protocol"] = Protocol([(1024, 4), (384, 4)], vocab_size=3)

        p = state["protocol"]
        p.eval()
        state["request_count"] += 1

        views_a = [torch.randn(1, 4, cfg[0]) for cfg in [(1024, 4), (384, 4)]]
        views_b = [torch.randn(1, 4, cfg[0]) for cfg in [(1024, 4), (384, 4)]]

        t0 = time.perf_counter()
        with torch.no_grad():
            msg_a, logits_a = p.encode(views_a)
            msg_b, logits_b = p.encode(views_b)
            pred = p.receivers[0](msg_a, msg_b).item()
        latency = (time.perf_counter() - t0) * 1000

        tokens_a = [l.argmax(-1).item() for l in logits_a]
        tokens_b = [l.argmax(-1).item() for l in logits_b]

        return {
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
            "prediction": float(pred),
            "a_greater": pred > 0,
            "latency_ms": latency,
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        info = {
            "version": "0.1.0",
            "protocol_loaded": state["protocol"] is not None,
            "requests": state["request_count"],
        }
        return generate_health_report(state["metrics_history"], info)

    return app


# Module-level app for uvicorn
if HAS_FASTAPI:
    app = create_app()


if __name__ == "__main__":
    if not HAS_FASTAPI:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        print("Then: uvicorn wmcp.server:app --port 8000")
    else:
        import uvicorn
        uvicorn.run("wmcp.server:app", host="0.0.0.0", port=8000, reload=True)
