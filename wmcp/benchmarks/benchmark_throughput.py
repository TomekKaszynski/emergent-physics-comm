"""Max throughput benchmark under load."""

import time, json, threading, queue
import numpy as np
import torch
from wmcp.protocol import Protocol


def run(n_messages: int = 10000, batch_sizes=None, output_path: str = None) -> dict:
    """Benchmark max messages/second under load.

    Args:
        n_messages: Total messages to process.
        batch_sizes: List of batch sizes to test.
        output_path: Optional JSON output path.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 32, 64]

    protocol = Protocol([(1024, 4), (384, 4)], vocab_size=3)
    protocol.eval()

    results = {}
    for bs in batch_sizes:
        views = [torch.randn(bs, 4, 1024), torch.randn(bs, 4, 384)]
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                protocol.communicate(views, views)

        n_batches = max(1, n_messages // bs)
        t0 = time.perf_counter()
        for _ in range(n_batches):
            with torch.no_grad():
                protocol.communicate(views, views)
        elapsed = time.perf_counter() - t0

        total = n_batches * bs
        results[str(bs)] = {
            "batch_size": bs,
            "total_messages": total,
            "elapsed_s": float(elapsed),
            "throughput_per_s": float(total / elapsed),
        }

    result = {"benchmark": "throughput", "results": results}
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = run(n_messages=5000)
    for bs, data in r["results"].items():
        print(f"  batch={bs}: {data['throughput_per_s']:.0f} msgs/s")
