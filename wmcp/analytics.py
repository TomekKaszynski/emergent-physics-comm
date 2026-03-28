"""Protocol analytics export — CSV, JSON, Prometheus-compatible formats."""

import json
import time
import io
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ProtocolMetrics:
    """Snapshot of protocol metrics at a point in time."""
    timestamp: float
    n_agents: int
    total_messages: int
    accuracy: float
    posdis: float
    topsim: float
    bosdis: float
    latency_mean_ms: float
    latency_p95_ms: float
    entropy_mean: float
    drift_detected: bool = False
    adversarial_agents: int = 0


def export_csv(metrics: List[ProtocolMetrics]) -> str:
    """Export metrics to CSV format."""
    lines = ["timestamp,n_agents,total_messages,accuracy,posdis,topsim,bosdis,"
             "latency_mean_ms,latency_p95_ms,entropy_mean,drift_detected,adversarial_agents"]
    for m in metrics:
        lines.append(f"{m.timestamp},{m.n_agents},{m.total_messages},"
                     f"{m.accuracy:.4f},{m.posdis:.4f},{m.topsim:.4f},{m.bosdis:.4f},"
                     f"{m.latency_mean_ms:.2f},{m.latency_p95_ms:.2f},"
                     f"{m.entropy_mean:.4f},{int(m.drift_detected)},{m.adversarial_agents}")
    return "\n".join(lines)


def export_json(metrics: List[ProtocolMetrics]) -> str:
    """Export metrics to JSON format."""
    return json.dumps([{
        "timestamp": m.timestamp,
        "n_agents": m.n_agents,
        "total_messages": m.total_messages,
        "accuracy": m.accuracy,
        "posdis": m.posdis,
        "topsim": m.topsim,
        "bosdis": m.bosdis,
        "latency_mean_ms": m.latency_mean_ms,
        "latency_p95_ms": m.latency_p95_ms,
        "entropy_mean": m.entropy_mean,
        "drift_detected": m.drift_detected,
        "adversarial_agents": m.adversarial_agents,
    } for m in metrics], indent=2)


def export_prometheus(metrics: ProtocolMetrics, prefix: str = "wmcp") -> str:
    """Export current metrics in Prometheus exposition format."""
    lines = [
        f"# HELP {prefix}_agents_total Number of active agents",
        f"# TYPE {prefix}_agents_total gauge",
        f"{prefix}_agents_total {metrics.n_agents}",
        f"# HELP {prefix}_messages_total Total messages processed",
        f"# TYPE {prefix}_messages_total counter",
        f"{prefix}_messages_total {metrics.total_messages}",
        f"# HELP {prefix}_accuracy Protocol accuracy on holdout",
        f"# TYPE {prefix}_accuracy gauge",
        f"{prefix}_accuracy {metrics.accuracy:.4f}",
        f"# HELP {prefix}_posdis Positional disentanglement score",
        f"# TYPE {prefix}_posdis gauge",
        f"{prefix}_posdis {metrics.posdis:.4f}",
        f"# HELP {prefix}_latency_mean_ms Mean round-trip latency",
        f"# TYPE {prefix}_latency_mean_ms gauge",
        f"{prefix}_latency_mean_ms {metrics.latency_mean_ms:.2f}",
        f"# HELP {prefix}_latency_p95_ms P95 round-trip latency",
        f"# TYPE {prefix}_latency_p95_ms gauge",
        f"{prefix}_latency_p95_ms {metrics.latency_p95_ms:.2f}",
        f"# HELP {prefix}_drift_detected Whether distribution drift was detected",
        f"# TYPE {prefix}_drift_detected gauge",
        f"{prefix}_drift_detected {int(metrics.drift_detected)}",
    ]
    return "\n".join(lines)


def generate_health_report(metrics: List[ProtocolMetrics],
                           protocol_info: Dict) -> str:
    """Generate an HTML health report."""
    if not metrics:
        return "<html><body><h1>No metrics available</h1></body></html>"

    latest = metrics[-1]
    status_color = "#22c55e" if latest.accuracy > 0.7 and not latest.drift_detected else "#ef4444"

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>WMCP Health Report</title>
<style>body{{font-family:sans-serif;max-width:800px;margin:0 auto;padding:20px}}
.metric{{display:inline-block;background:#f5f5f5;padding:16px;margin:8px;border-radius:8px;min-width:150px;text-align:center}}
.metric .value{{font-size:28px;font-weight:bold;color:#1a1a1a}}.metric .label{{font-size:12px;color:#666}}
.status{{display:inline-block;padding:4px 12px;border-radius:12px;color:white;font-weight:bold;background:{status_color}}}
</style></head><body>
<h1>WMCP Protocol Health Report</h1>
<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<p>Status: <span class="status">{'HEALTHY' if status_color == '#22c55e' else 'DEGRADED'}</span></p>
<div>
<div class="metric"><div class="value">{latest.n_agents}</div><div class="label">Active Agents</div></div>
<div class="metric"><div class="value">{latest.total_messages:,}</div><div class="label">Total Messages</div></div>
<div class="metric"><div class="value">{latest.accuracy:.1%}</div><div class="label">Accuracy</div></div>
<div class="metric"><div class="value">{latest.posdis:.3f}</div><div class="label">PosDis</div></div>
<div class="metric"><div class="value">{latest.latency_mean_ms:.1f}ms</div><div class="label">Latency</div></div>
<div class="metric"><div class="value">{latest.adversarial_agents}</div><div class="label">Adversarial</div></div>
</div>
<h2>Protocol Info</h2>
<pre>{json.dumps(protocol_info, indent=2)}</pre>
</body></html>"""
    return html
