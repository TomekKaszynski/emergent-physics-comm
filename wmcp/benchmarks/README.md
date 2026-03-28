# WMCP Benchmark Suite

Standardized benchmarks for evaluating WMCP protocol performance.

## Running Benchmarks

```bash
# Individual benchmarks
python -m wmcp.benchmarks.benchmark_latency
python -m wmcp.benchmarks.benchmark_throughput
python -m wmcp.benchmarks.benchmark_compression
python -m wmcp.benchmarks.benchmark_scaling
python -m wmcp.benchmarks.benchmark_onboarding

# Save results to JSON
python -c "from wmcp.benchmarks.benchmark_latency import run; run(output_path='latency.json')"
```

## Benchmarks

| Benchmark | Measures | Key Metric |
|-----------|----------|------------|
| `benchmark_latency` | Single-sample round-trip time | Mean, P50, P95, P99 in ms |
| `benchmark_throughput` | Max messages/second under load | Throughput at batch sizes 1–64 |
| `benchmark_compression` | Bits per message vs raw features | Compression ratio per encoder |
| `benchmark_scaling` | PosDis vs population size | PosDis at 1–16 agents |
| `benchmark_onboarding` | Steps to onboard new encoder | Steps to 90% accuracy |

## Interpreting Results

- **Latency < 10ms**: Suitable for real-time robotics (Phase 103: 1.19ms CPU)
- **Throughput > 1000/s**: Sufficient for fleet-scale deployment (Phase 103: 4095/s batch)
- **Compression > 1000×**: Significant bandwidth savings (Phase 113: 5200×)
- **Scaling PosDis > 0.5**: Compositionality maintained at scale (Phase 99: 0.534 at 16 agents)
- **Onboarding < 100 steps**: Fast new model integration (Phase 104: 50 steps)
