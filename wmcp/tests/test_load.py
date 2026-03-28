"""Load testing for WMCP protocol operations."""

import time
import json
import asyncio
import numpy as np
import torch
import threading
from pathlib import Path
from wmcp.protocol import Protocol


def test_concurrent_communication(n_requests: int = 1000, n_threads: int = 8):
    """Test concurrent communication requests.

    Args:
        n_requests: Total requests to process.
        n_threads: Number of concurrent threads.

    Returns:
        Dict with throughput and latency stats.
    """
    protocol = Protocol([(1024, 4), (384, 4)], vocab_size=3)
    protocol.eval()

    latencies = []
    errors = [0]
    lock = threading.Lock()

    def worker(n_per_thread):
        for _ in range(n_per_thread):
            views_a = [torch.randn(1, 4, 1024), torch.randn(1, 4, 384)]
            views_b = [torch.randn(1, 4, 1024), torch.randn(1, 4, 384)]
            try:
                t0 = time.perf_counter()
                with torch.no_grad():
                    protocol.communicate(views_a, views_b)
                lat = (time.perf_counter() - t0) * 1000
                with lock:
                    latencies.append(lat)
            except Exception:
                with lock:
                    errors[0] += 1

    per_thread = n_requests // n_threads
    t_start = time.time()
    threads = [threading.Thread(target=worker, args=(per_thread,)) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = time.time() - t_start

    lats = np.array(latencies) if latencies else np.array([0])
    return {
        "test": "concurrent_communication",
        "n_requests": n_requests,
        "n_threads": n_threads,
        "completed": len(latencies),
        "errors": errors[0],
        "error_rate": errors[0] / n_requests,
        "total_time_s": total_time,
        "requests_per_s": len(latencies) / total_time,
        "latency_p50_ms": float(np.percentile(lats, 50)),
        "latency_p95_ms": float(np.percentile(lats, 95)),
        "latency_p99_ms": float(np.percentile(lats, 99)),
    }


def test_memory_under_load(n_rounds: int = 5000):
    """Test memory stability under sustained load.

    Returns memory usage at checkpoints.
    """
    import os

    protocol = Protocol([(1024, 4), (384, 4)], vocab_size=3)
    protocol.eval()

    checkpoints = []
    for i in range(n_rounds):
        views = [torch.randn(1, 4, 1024), torch.randn(1, 4, 384)]
        with torch.no_grad():
            protocol.communicate(views, views)

        if (i + 1) % 1000 == 0:
            # RSS memory (rough)
            try:
                import resource
                mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
            except Exception:
                mem_mb = 0
            checkpoints.append({"round": i + 1, "mem_mb": mem_mb})

    return {"test": "memory_stability", "checkpoints": checkpoints}


def run_load_tests(output_path: str = None):
    """Run all load tests and save results."""
    print("WMCP Load Tests", flush=True)
    print("=" * 40, flush=True)

    results = {}

    print("  Concurrent communication (1000 req, 8 threads)...", flush=True)
    r = test_concurrent_communication(1000, 8)
    results["concurrent"] = r
    print(f"    {r['requests_per_s']:.0f} req/s, "
          f"p50={r['latency_p50_ms']:.1f}ms, "
          f"p99={r['latency_p99_ms']:.1f}ms, "
          f"errors={r['errors']}", flush=True)

    print("  Memory stability (5000 rounds)...", flush=True)
    r2 = test_memory_under_load(5000)
    results["memory"] = r2
    if r2["checkpoints"]:
        print(f"    Memory at 5000 rounds: {r2['checkpoints'][-1]['mem_mb']:.0f}MB", flush=True)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved {output_path}", flush=True)

    return results


if __name__ == "__main__":
    run_load_tests("results/load_test_results.json")
