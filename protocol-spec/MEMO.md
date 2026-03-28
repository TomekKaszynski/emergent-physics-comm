# WMCP: World Model Communication Protocol

## Problem

$4.3B has been invested in vision foundation models (V-JEPA, DINOv2, CLIP, SAM, Gemini). Each produces incompatible latent representations. Multi-model robotics systems — where different sensors run different models — cannot communicate about physical scene properties. Linear alignment fails across domains (R² < 0 on cross-scenario transfer). There is no protocol layer for world models.

## Solution

A discrete bottleneck (K=3 symbols per message position) forces heterogeneous vision encoders to develop shared compositional communication during co-training. No alignment maps. No shared architecture. The bottleneck itself is the protocol. New models join by fine-tuning a lightweight projection layer (~400K parameters) while existing agents remain frozen. The protocol compresses encoder features by 5,200× while preserving task-relevant physical properties.

## Evidence

118 experiment phases. 1,350+ training runs across 3 physics scenarios (spring, fall, ramp). Three encoder architectures validated: V-JEPA 2 (temporal SSL), DINOv2 (spatial SSL), CLIP ViT-L/14 (language-supervised). Key results:

- **1.19ms CPU latency** (under 10ms robotics threshold)
- **50 training steps** to onboard a new encoder to 90% accuracy
- **886K minimum parameters** for a passing projection layer
- **22 seconds** to bootstrap a new domain protocol from scratch
- **100-agent stress test**, zero message drops
- **81.8% accuracy** on real video (MIT Physics 101), heterogeneous agents
- **Three compositionality metrics** (PosDis, TopSim, BosDis) converge with zero divergences
- **5,200× bandwidth compression** vs raw feature transmission
- Validated on real camera footage, not just simulation

## 90-Day Plan

1. **Protocol spec v0.2** — reference PyTorch implementation (wmcp package exists, 23 tests pass)
2. **3 external reproductions** — ship Colab notebook, collect independent validation
3. **ROS 2 integration demo** — wrap trained agents as ROS 2 nodes, demonstrate on real robot
4. **NeurIPS 2026 submission** — paper drafted, 118 phases of experimental evidence
5. **HuggingFace Hub** — publish model cards and pre-trained protocol instances

## Why Me

20-year-old independent researcher. Built the entire system — 118 experiment phases in 6 weeks on a MacBook M3 Pro, no institutional affiliation, no formal ML training. AI-orchestrated methodology: Claude Code managed the experiment pipeline, I directed the research questions. The protocol emerged from a genuine research insight about discrete bottlenecks, validated through systematic empirical work at a scale that exceeds most PhD dissertations.

---

*Tomek Kaszynski · [t.kaszynski@proton.me](mailto:t.kaszynski@proton.me) · [doi:10.5281/zenodo.19197757](https://doi.org/10.5281/zenodo.19197757)*
