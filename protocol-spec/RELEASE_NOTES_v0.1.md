# WMCP v0.1.0 Release Notes

**Release date:** March 28, 2026

## Summary

First public release of the World Model Communication Protocol. WMCP v0.1.0 establishes that heterogeneous vision foundation models can develop shared compositional communication through a discrete bottleneck, without alignment maps or shared architecture.

## What's in this release

**A protocol specification** — message format (K=3 symbols, L=2 positions), encoding procedure (frozen encoder → projection → Gumbel-Softmax), compositionality requirements (PosDis > 0.5), and robustness profile.

**A pip-installable Python package** — `wmcp` with 23 unit tests, CLI, FastAPI server, monitoring, benchmarks, and visualization tools.

**148 experiment phases** — 1,350+ training runs across 3 physics scenarios, 3 encoder architectures, population sizes from 1 to 16 agents, validated on real camera footage from MIT Physics 101.

## Key numbers

- **1.19ms** CPU inference latency
- **50 steps** to onboard a new encoder
- **5,200×** compression vs raw features
- **3 architectures** validated (V-JEPA 2, DINOv2, CLIP ViT-L/14)
- **23 tests** pass in clean venv
- **0 message drops** in 100-agent stress test

## Known limitations

- Physics properties only (mass, elasticity, friction)
- Co-training required for new encoders (not plug-and-play)
- No cross-domain transfer (each scenario needs its own protocol)
- Maximum validated population: 16 agents
- Real-video validation on Physics 101 only

## What's next (v0.2)

- Reference implementation cleanup
- External reproduction (Colab notebook shipped)
- ROS 2 integration hardening
- NeurIPS 2026 submission
- HuggingFace Hub publication
