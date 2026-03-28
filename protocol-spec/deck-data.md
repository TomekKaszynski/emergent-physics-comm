# WMCP Deck Data — Key Numbers

Copy-paste ready for presentation slides.

## Headline
**WMCP: World Model Communication Protocol**
Discrete compositional communication between heterogeneous vision foundation models.

## Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Validated architectures | 3 (V-JEPA 2, DINOv2, CLIP ViT-L/14) | Phase 96 |
| Total experiment phases | 115 | Phases 1–115 |
| Total training runs | 1,350+ (Phase 94 sweep) + 500+ (other phases) | — |
| CPU inference latency | 1.19ms | Phase 103 |
| Pub-sub latency | 0.70ms | Phase 106 |
| New model onboarding | 50 steps to 90% accuracy | Phase 104 |
| Minimum projection layer | 886K parameters (hidden_dim=8) | Phase 108 |
| Domain bootstrap | 22 seconds (spring) | Phase 109 |
| Stress test | 100 agents, 0 message drops | Phase 107 |
| Noise tolerance | Graceful to σ=0.9 (18pp above chance) | Phase 98 |
| Real-video accuracy | 81.8% hetero (Physics 101 spring) | Phase 95 |
| Triple metric agreement | PosDis, TopSim, BosDis — zero divergences | Phase 94 |
| Compliance suite | 8/9 tests pass | Phase 103 compliance |
| Population scaling | 1–16 agents, PosDis never collapses | Phase 99 |
| Codebook sweet spot | K=3 (86% cross-arch token agreement) | Phase 92c |
| Compression ratio | 5,200× vs raw features | Phase 113 |
| INT8 quantization | <2% accuracy drop (projected) | Phase 111 |

## One-Liners for Slides

- "Three vision architectures. One protocol. Zero alignment maps."
- "50 training steps to add a new model. 22 seconds to bootstrap a new domain."
- "1.19ms latency on CPU. Real-time ready."
- "1,350 runs. Three metrics. Zero divergences."
- "The discrete bottleneck IS the protocol layer."