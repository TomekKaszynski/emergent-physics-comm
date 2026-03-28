# Emergent Compositional Communication for Latent World Properties

**Tomek Kaszynski** · [t.kaszynski@proton.me](mailto:t.kaszynski@proton.me)

> Neural agents with different vision backbones develop shared compositional languages about physics through a discrete bottleneck — no alignment maps, no shared architecture.

[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.19197757)
[![Protocol Spec](https://img.shields.io/badge/WMCP-v0.1_Draft-orange)](protocol-spec/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

## Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Experiment phases | 109 | Phases 1–109 |
| Architecture pairings validated | V-JEPA 2 + DINOv2 + CLIP ViT-L/14 | Phase 96 |
| Total training runs | 1,350+ (full sweep) | Phase 94 |
| Compositionality metrics | 3 (PosDis, TopSim, BosDis), zero divergences | Phase 94 |
| Real-video accuracy (hetero) | 81.8% on Physics 101 spring | Phase 95 |
| Inference latency | 1.19ms CPU, 7.88ms MPS | Phase 103 |
| New model onboarding | 50 training steps | Phase 104 |
| Population scaling | 1–16 agents, PosDis never collapses | Phase 99 |
| Noise tolerance | <6pp drop at σ=0.5 | Phase 98 |

## What This Does

Three vision models with completely different training objectives — **V-JEPA 2** (self-supervised video prediction), **DINOv2** (self-supervised self-distillation), and **CLIP ViT-L/14** (language-supervised contrastive) — develop compositional communication protocols about physical properties (mass, elasticity, friction) when trained together through a discrete bottleneck.

Each agent independently learns to map high-dimensional visual features into 3 discrete symbols per message position. Without any explicit coordination, message positions self-organize to encode distinct physical attributes. The protocol works on both synthetic physics simulations and real camera footage (MIT Physics 101 dataset).

## Protocol Specification

The protocol is formalized as **WMCP (World Model Communication Protocol) v0.1**:

→ **[protocol-spec/](protocol-spec/)** — Full specification, architecture diagrams, onboarding guide, evidence summary

## Quick Start

Reproduce the core result (heterogeneous V-JEPA+DINOv2 emergent communication on real video) in ~3 minutes:

```bash
# Prerequisites: torch>=2.0, numpy, scipy, transformers>=4.40
# Requires pre-extracted features in results/ (see Phase 87)

# Run heterogeneous 2-agent communication on Physics 101 spring
PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "
from _phase95_realvideo import run_phase95
run_phase95()
"
```

Or run the onboarding demo (add CLIP to an existing protocol):

```bash
# Full demo (~5 min)
python3 protocol-spec/examples/onboard_new_encoder.py

# Cached results (instant)
python3 protocol-spec/examples/onboard_new_encoder.py --dry-run
```

## Results (Phase 94: Triple Metrics, 1,350 Runs)

| Scenario | Pairing | PosDis | TopSim | BosDis | Accuracy |
|----------|---------|--------|--------|--------|----------|
| Spring | Heterogeneous | 0.669 | 0.425 | 0.613 | 78.7% |
| Spring | V-JEPA homo | 0.681 | 0.465 | 0.619 | 79.6% |
| Spring | DINOv2 homo | 0.566 | 0.359 | 0.554 | 72.3% |
| Fall | Heterogeneous | 0.470 | 0.439 | 0.491 | 77.4% |
| Fall | V-JEPA homo | 0.542 | 0.481 | 0.517 | 78.5% |
| Ramp | Heterogeneous | 0.424 | 0.305 | 0.465 | 73.5% |
| Ramp | DINOv2 homo | 0.508 | 0.349 | 0.502 | 72.7% |

All three compositionality metrics agree across 1,350 runs with zero divergences. Heterogeneous agents match homogeneous performance — the discrete bottleneck forces compatible protocols regardless of encoder architecture.

## Three-Architecture Validation (Phase 96)

| Pairing | PosDis | Accuracy |
|---------|--------|----------|
| V-JEPA + DINOv2 | 0.764 | 81.8% |
| V-JEPA + CLIP | 0.737 | 75.7% |
| DINOv2 + CLIP | 0.657 | 70.2% |
| V-JEPA + DINOv2 + CLIP | 0.764 | 81.8% |

## Repository Structure

```
├── protocol-spec/              # WMCP v0.1 protocol specification
│   ├── SPEC.md                 # Core specification
│   ├── ARCHITECTURE.md         # System diagrams (Mermaid)
│   ├── ONBOARDING.md           # How to add new encoders
│   ├── EVIDENCE.md             # Experimental evidence summary
│   ├── examples/               # Demo scripts
│   └── tests/                  # Compliance tests
├── physics_sim.py              # Physics environments + data generation
├── world_model.py              # Neural networks
├── run_all.py                  # Experiment launcher
├── _phase*.py                  # Individual experiment scripts (Phases 1–109)
├── EXPERIMENTS.md              # Full experiment log
└── results/                    # Metrics, visualizations, checkpoints
```

## Method

1. **Visual encoding**: Frozen V-JEPA 2, DINOv2, or CLIP features from physics videos
2. **Multi-agent communication**: Each agent observes a temporal window and sends a discrete message via Gumbel-Softmax (K=3 symbols per position)
3. **Pairwise comparison task**: A receiver predicts which of two scenes has higher values for each physical property
4. **Population pressure**: Receiver reset every 40 epochs with 3 simultaneous receivers drives compositionality

## Requirements

```
torch>=2.0
torchvision
numpy
scipy
matplotlib
transformers>=4.40    # V-JEPA 2 features
open-clip-torch       # CLIP features
```

Hardware: Tested on M3 MacBook Pro with MPS acceleration (float32 only).

## Citation

```bibtex
@article{kaszynski2026emergent,
  title={Emergent Compositional Communication for Latent World Properties},
  author={Kaszy{\'n}ski, Tomek},
  year={2026},
  doi={10.5281/zenodo.19197757}
}
```

## License

MIT
