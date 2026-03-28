"""Standardized latency benchmark for WMCP protocol."""

import time
import json
import numpy as np
import torch
from wmcp.protocol import Protocol


def run(n_iterations: int = 1000, configs=None, output_path: str = None) -> dict:
    """Run latency benchmark.

    Args:
        n_iterations: Number of communication rounds.
        configs: Agent configs, default [(1024,4),(384,4)].
        output_path: Optional JSON output path.

    Returns:
        Dict with latency statistics.
    """
    if configs is None:
        configs = [(1024, 4), (384, 4)]

    protocol = Protocol(configs, vocab_size=3)
    protocol.eval()

    views = [torch.randn(1, nf, d) for d, nf in configs]

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            protocol.communicate(views, views)

    latencies = []
    for _ in range(n_iterations):
        t = time.perf_counter()
        with torch.no_grad():
            protocol.communicate(views, views)
        latencies.append((time.perf_counter() - t) * 1000)

    lats = np.array(latencies)
    result = {
        "benchmark": "latency",
        "n_iterations": n_iterations,
        "n_agents": len(configs),
        "mean_ms": float(np.mean(lats)),
        "median_ms": float(np.median(lats)),
        "p95_ms": float(np.percentile(lats, 95)),
        "p99_ms": float(np.percentile(lats, 99)),
        "min_ms": float(np.min(lats)),
        "max_ms": float(np.max(lats)),
        "throughput_per_s": float(1000 / np.mean(lats)),
        "realtime_viable": float(np.mean(lats)) < 10.0,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    r = run()
    print(f"Latency: mean={r['mean_ms']:.2f}ms p95={r['p95_ms']:.2f}ms "
          f"throughput={r['throughput_per_s']:.0f}/s "
          f"realtime={'YES' if r['realtime_viable'] else 'NO'}")
