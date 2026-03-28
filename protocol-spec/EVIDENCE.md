# WMCP Experimental Evidence

Summary of experimental evidence supporting each specification claim. All experiments use the Physics 101 dataset (spring, fall, ramp scenarios) with frozen V-JEPA 2, DINOv2, and CLIP ViT-L/14 encoders. Experiment code: [github.com/TomekKaszynski/emergent-physics-comm](https://github.com/TomekKaszynski/emergent-physics-comm).

## Architecture Agnosticism

**Claim:** The protocol works across fundamentally different vision architectures.

| Phase | Finding | Key Metric |
|-------|---------|------------|
| Phase 93 | Heterogeneous V-JEPA+DINOv2 agents develop compositional protocols from scratch — no alignment needed | PosDis 0.824 (K=3, 2-agent, 5 seeds) |
| Phase 94 | 1,350-run sweep confirms hetero agents match homo performance across 3 scenarios × 5 codebook sizes × 3 population sizes | Hetero PosDis 0.521 vs HomoVV 0.541, gap = −0.020 |
| Phase 96 | Three architectures (V-JEPA 2, DINOv2, CLIP ViT-L/14) all develop compatible protocols | All pairwise PosDis > 0.5; 3-arch pool PosDis 0.764 |

**Interpretation:** The discrete bottleneck forces shared structure regardless of encoder training objective (temporal SSL, spatial SSL, language-supervised contrastive). Architecture diversity acts as a regularizer — heterogeneous populations achieve +0.044 PosDis over homogeneous on average.

## Real-Video Validation

**Claim:** The protocol works on real camera footage, not just synthetic data.

| Phase | Finding | Key Metric |
|-------|---------|------------|
| Phase 87 | V-JEPA 2 features from Physics 101 videos encode mass (pairwise AUC 0.905) | AUC 0.905 ± 0.05 |
| Phase 95 | Hetero V-JEPA+DINOv2 achieves 81.8% accuracy on real video mass comparison | 81.8% ± 7.3% (10 seeds) |
| Phase 95 | Both agents independently encode mass with perfect monotonic correlation | Spearman ρ = 1.000 for both agents |
| Phase 101 | Real-video validation extends to all three Physics 101 scenarios | Spring 81.8%, Fall 86.7%, Ramp 82.1% |

**Interpretation:** The protocol generalizes from synthetic environments to real laboratory footage. Mass comparison accuracy on real video matches synthetic baselines, and per-agent mass encoding is monotonic — each agent independently learns to map physical properties to discrete symbols.

## Compositionality

**Claim:** Messages exhibit genuine compositional structure, verified by three independent metrics.

| Phase | Finding | Key Metric |
|-------|---------|------------|
| Phase 94 | Triple metrics (PosDis, TopSim, BosDis) converge with zero divergences across 1,350 runs | 0 divergences flagged |
| Phase 94 | Compositionality emerges in all conditions: all 3 pairings × 5 K values × 3 population sizes | No condition collapses |
| Phase 102 | MI heatmaps show clear position-attribute specialization; mass dominates all positions | MI(position, mass) = 0.4–0.7 |
| Phase 93 | At K=3, 100% of message positions show monotonic mass encoding (4/4 positions) | Monotonic = 4/4 (all 5 seeds) |

**Interpretation:** Compositionality is not an artifact of a single metric. Three conceptually different measures — position specialization (PosDis), meaning-message correlation (TopSim), and symbol-attribute mapping (BosDis) — agree across all conditions. The MI matrix confirms each position preferentially encodes mass over object identity.

## Noise Robustness

**Claim:** The protocol degrades gracefully under message noise.

| Phase | Finding | Key Metric |
|-------|---------|------------|
| Phase 98 | Gaussian noise σ = 0.5 causes < 6pp accuracy drop | 72.1% at σ=0.5 vs 77.9% baseline |
| Phase 98 | At σ = 0.9, accuracy remains 18pp above chance | 67.8% at σ=0.9 (chance = 50%) |

**Interpretation:** The discrete one-hot encoding creates inherent noise tolerance. Moderate noise (σ ≤ 0.3) has negligible effect (< 3pp). Even extreme noise does not cause catastrophic failure — the receiver extracts partial information from noisy messages.

## Population Scaling

**Claim:** The protocol scales from 1 to 16 agents without collapse.

| Phase | Finding | Key Metric |
|-------|---------|------------|
| Phase 99 | PosDis decreases monotonically but stays above 0.5 at 16 agents | Hetero: 0.788 (n=1) → 0.534 (n=16) |
| Phase 99 | HomoVV degrades slower than HomoDD; DINOv2-only reaches 0.378 at n=16 | HomoDD PosDis 0.378 at n=16 |

**Interpretation:** More agents create more message positions, increasing the difficulty of per-position specialization. The decrease is expected and gradual — compositionality persists. V-JEPA provides more robust scaling due to richer temporal physics features.

## Co-Training Requirement

**Claim:** Zero-shot transfer fails; co-training through the bottleneck is required.

| Phase | Finding | Key Metric |
|-------|---------|------------|
| Phase 97 | Swapping independently-trained senders drops accuracy 15–29pp | vjepa→dino: −19.1pp; vjepa→clip: −28.9pp |
| Phase 104 | Co-training a new CLIP agent reaches 90% of base in 50 steps | 10/10 seeds converged; final acc 82.8% |
| Phase 91 | Linear alignment R² = 0.068 (cross-validated); alignment maps are fragile | R² = 0.068 |

**Interpretation:** Independently trained encoders produce geometrically incompatible representations (Phase 91). The protocol's value is that it forces compatibility during co-training — the discrete bottleneck acts as a coordination mechanism. Onboarding is fast (50 steps) because the bottleneck constrains the solution space.

## Domain Specificity

**Claim:** Each physics domain requires its own protocol instance.

| Phase | Finding | Key Metric |
|-------|---------|------------|
| Phase 100 | Spring→ramp transfer accuracy is at chance | 51.8% ± 4.4% (chance = 50%) |
| Phase 100 | Spring→fall transfer is slightly above chance | 56.5% ± 8.3% (hetero) |
| Phase 92b | Spring-trained alignment produces R² < −4 on fall/ramp | R²(V→D) = −6.37 on fall |

**Interpretation:** The protocol encodes domain-specific physical relationships. Spring elasticity patterns do not predict fall restitution or ramp friction. This is correct behavior — different physics require different communication. The protocol structure (K, L, architecture) transfers; only the learned weights are domain-specific.

## Heterogeneous Advantage

**Claim:** Mixed-architecture populations can outperform same-architecture populations.

| Phase | Finding | Key Metric |
|-------|---------|------------|
| Phase 96 | Hetero populations average +0.044 PosDis over homo | Mean across 8 hetero vs 6 homo conditions |
| Phase 101 | On ramp, hetero (82.1%) beats HomoVV (75.7%) | +6.4pp accuracy advantage |
| Phase 93 | At K=3, hetero achieves highest PosDis of any condition (0.824) | 5 seeds, 2-agent |

**Interpretation:** Architectural diversity acts as a regularizer. When agents with different input representations must communicate through a shared bottleneck, they are forced toward more systematic (compositional) encoding. The advantage is most pronounced at small codebook sizes (K=3) where the bottleneck pressure is strongest. On ramp scenarios, DINOv2's appearance features complement V-JEPA's temporal features, providing information neither architecture has alone.

## Latency and Deployment

**Claim:** The protocol is viable for real-time systems.

| Phase | Finding | Key Metric |
|-------|---------|------------|
| Phase 103 | CPU round-trip latency: 1.19ms mean | Under 10ms robotics threshold |
| Phase 103 | Batch throughput: 4,095 comms/s on MPS | Suitable for multi-agent deployment |
| Phase 106 | Async pub-sub PoC: 0.70ms latency, 1,424 comms/s, 92.9% accuracy | End-to-end system validation |

**Interpretation:** The protocol's lightweight architecture (400K params per agent) enables sub-millisecond inference on CPU. The pub-sub integration demonstrates the protocol works as a real message-passing system, not just a training-time abstraction.

## Optimal Configuration

**Claim:** K=3 is the optimal codebook size for cross-architecture communication.

| Phase | Finding | Key Metric |
|-------|---------|------------|
| Phase 92c | K=3 achieves highest cross-architecture token agreement (86.0%) and transferred PosDis (0.803) | Agreement 86.0%, PosDis 0.803 |
| Phase 92c | Larger codebooks degrade agreement monotonically | K=32 agreement: 67.0% |
| Phase 93 | K=3 is the only codebook size where hetero beats both homo baselines | Hetero 0.824 vs HomoVV 0.764 |

**Interpretation:** Smaller codebooks create stronger information bottleneck pressure, which forces more systematic compositional structure. K=3 (3 symbols, 9 possible messages for 2 positions) is the sweet spot — sufficient capacity for mass discrimination, minimal fragility to alignment noise.
