# WMCP Compliance Report

Generated: 2026-03-28 12:04:58

**Result: 8/9 tests PASSED**

| # | Requirement | Status | Detail |
|---|-------------|--------|--------|
| 1 | Message format: correct dimensions | PASS | Expected 24, got 24 |
| 2 | Message format: one-hot per position | PASS | Checked 8 positions |
| 3 | Compositionality: PosDis > 0.5 | PASS | PosDis = 0.535 |
| 4 | Noise tolerance: <10% drop at σ=0.5 | FAIL | Baseline=86.5%, Noisy=70.4%, Drop=16.1% |
| 5 | Latency: <10ms on CPU | PASS | Mean=1.11ms, P95=1.13ms |
| 6 | Config: K=3 vocabulary | PASS | K=3 |
| 7 | Config: heterogeneous 4-agent | PASS | n_agents=4, dims={1024, 384} |
| 8 | MI structure: mass > object identity | PASS | Mass MI=0.436, Object MI=0.123 |
| 9 | Task performance: holdout acc > 60% | PASS | Accuracy=86.9% |

## Configuration

- K = 3
- L = 2 positions per agent
- N = 4 agents
- Architecture: heterogeneous (V-JEPA 2 + DINOv2)
- Dataset: Physics 101 spring (206 clips, 26 objects)
- Spec version: WMCP v0.1