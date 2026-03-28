# WMCP Specification v0.1

## 1. Overview

### Problem
Heterogeneous vision foundation models encode scenes into incompatible latent spaces. A V-JEPA 2 model's 1024-dimensional temporal features cannot be directly compared, fused, or communicated with a DINOv2 model's 384-dimensional spatial features or a CLIP model's 768-dimensional language-grounded features. Cross-architecture alignment via linear maps is fragile, scenario-specific, and degrades under distribution shift (Phase 91: cross-validated R² = 0.068; Phase 92b: spring-trained alignment produces negative R² on fall/ramp).

### Solution
WMCP defines a discrete compositional communication channel that forces heterogeneous encoders to develop shared symbolic representations during co-training. Each encoder learns a lightweight projection layer that maps its features into a shared discrete vocabulary. The discrete bottleneck — not the encoder architecture — determines the communication structure.

### Design Philosophy
The discrete bottleneck IS the protocol. Compositionality is not engineered — it emerges from the information constraint. When agents must compress high-dimensional continuous representations into a small set of discrete symbols to solve a cooperative task, they independently converge on compositional encodings where each symbol position specializes for a distinct physical property.

## 2. Message Format

### Vocabulary
Each message position selects one symbol from a vocabulary of K symbols. The optimal vocabulary size is **K = 3**, validated across all architecture pairings and population sizes (Phase 92c: K=3 achieves highest cross-architecture token agreement at 86.0% and transferred PosDis of 0.803).

Larger vocabularies (K = 5, 8, 16, 32) are supported but provide diminishing returns for compositionality and degrade cross-architecture agreement (Phase 92c: agreement drops from 86.0% at K=3 to 67.0% at K=32).

### Message Structure
A message consists of L symbol positions:

```
message = [s₁, s₂, ..., sₗ]    where sᵢ ∈ {0, 1, ..., K-1}
```

Each position encodes information about one physical property. With K = 3 and L = 2 positions per agent:
- Total message space: K^L = 9 possible messages per agent
- Information capacity: L × log₂(K) = 3.17 bits per agent

### Multi-Agent Messages
In a population of N agents, each agent produces an independent message. The joint message is the concatenation:

```
joint_message = [agent₁_msg, agent₂_msg, ..., agentₙ_msg]
```

Total joint message length: N × L positions, total capacity: N × L × log₂(K) bits.

### Position Semantics
Message positions are not pre-assigned to properties. During training, each position self-organizes to encode a specific physical attribute. This is verified post-training via the mutual information matrix MI(position, attribute).

## 3. Encoding Procedure

### Input
Frozen features from any vision backbone. The encoder must be frozen — only the projection layer is trained. Validated input dimensions:

| Encoder | Training Objective | Feature Dim | Type |
|---------|-------------------|-------------|------|
| V-JEPA 2 ViT-L | Self-supervised video prediction | 1024 | Temporal (8 frames) |
| DINOv2 ViT-S/14 | Self-supervised self-distillation | 384 | Static (single frame) |
| CLIP ViT-L/14 | Language-supervised contrastive | 768 | Static (single frame) |

For static encoders, the single-frame feature is replicated across the temporal dimension to match the expected input format.

### Architecture

```
frozen_features (B, T, D)
    → Conv1d temporal encoder (D → 256 → 128)
    → AdaptiveAvgPool1d → (B, 128)
    → Linear (128 → hidden_dim)
    → ReLU
    → Linear head per position (hidden_dim → K)
    → Gumbel-Softmax (training) or argmax (inference)
    → one-hot symbol (B, K) per position
    → concatenate → message (B, L × K)
```

The projection layer (temporal encoder + linear head) is the ONLY trainable component per encoder. Total trainable parameters per agent: approximately 400K (for hidden_dim = 128).

### Discretization
- **Training:** Gumbel-Softmax with temperature τ, annealed from τ = 3.0 to τ = 1.0 over the training period. Hard straight-through estimator enabled after 30 warmup epochs.
- **Inference:** Deterministic argmax over logits, producing hard one-hot symbols.

## 4. Alignment Procedure

### Co-Training (Required)
New models join the protocol by co-training their projection layer alongside existing protocol agents on a shared cooperative task. During onboarding:

1. Existing agents' weights are **frozen**.
2. Only the new agent's projection layer is trained.
3. Training signal comes from a pairwise comparison task mediated by a shared receiver.

Onboarding is fast: 10/10 seeds reached 90% of base accuracy within 50 training steps (Phase 104). This corresponds to seconds of compute on a single GPU.

### Zero-Shot Transfer Does Not Work
Independently trained senders produce incompatible messages. Swapping a sender trained on architecture A with one trained on architecture B drops accuracy by 15–29 percentage points (Phase 97). This is by design: the bottleneck forces shared structure only during joint training. The protocol is a coordination mechanism, not a representation-alignment tool.

### Alignment Maps Are Not Required
Linear alignment maps (ridge regression, CKA-based) between encoder spaces are fragile and scenario-specific (Phase 92b: spring-trained alignment produces R² < −4 on fall/ramp). WMCP eliminates the need for explicit alignment by making the discrete bottleneck the alignment mechanism itself.

## 5. Compositionality Requirements

### Certification Metrics
Protocol messages must be validated using three independent compositionality measures:

1. **PosDis (Positional Disentanglement):** For each message position, the gap between its highest and second-highest mutual information with any attribute, normalized by the highest. Range [0, 1]. Measures whether each position specializes for one property.

2. **TopSim (Topographic Similarity):** Spearman correlation between meaning distances (Manhattan distance in property space) and message distances (Hamming distance in symbol space). Range [−1, 1]. Measures whether similar meanings produce similar messages.

3. **BosDis (Bag-of-Symbols Disentanglement):** Like PosDis but position-independent — measures whether specific symbols (regardless of position) uniquely encode specific attributes. Range [0, 1].

### Certification Threshold
- **PosDis ≥ 0.5** on held-out validation (object-level holdout, not sample-level).
- TopSim and BosDis reported alongside PosDis for triangulation.

These three metrics converge with zero divergences across 1,350 validated runs (Phase 94). When one metric is high, the others are proportionally high — there are no cases of high PosDis with low TopSim, which would indicate measurement artifacts.

### Validation Protocol
Compositionality is evaluated on held-out objects (20% of unique objects withheld from training). This ensures the protocol generalizes to unseen physical instances, not just unseen samples of known objects.

## 6. Robustness Profile

### Noise Tolerance
Gaussian noise added to messages post-encoding degrades accuracy gracefully, not catastrophically (Phase 98, 4-agent heterogeneous, K = 3):

| Noise σ | Accuracy | Degradation |
|---------|----------|-------------|
| 0.0 | 77.9% | — |
| 0.3 | 75.4% | −2.5pp |
| 0.5 | 72.1% | −5.8pp |
| 0.9 | 67.8% | −10.1pp |

Even at σ = 0.9 (noise magnitude comparable to one-hot message values), accuracy remains 18 percentage points above chance. The discrete bottleneck creates inherently noise-robust codes.

### Inference Latency
Single-sample round-trip latency on Apple M3 Pro (Phase 103):

| Device | Mean | P95 | Throughput |
|--------|------|-----|------------|
| CPU | 1.19ms | 1.35ms | 842/s |
| MPS (GPU) | 7.88ms | 11.13ms | 127/s |
| MPS batch=32 | 7.81ms | — | 4,095/s |

CPU inference (1.19ms) meets the <10ms requirement for real-time robotics applications.

## 7. Scalability

### Population Size
Validated from 1 to 16 agents (Phase 99, K = 3, spring scenario):

| N Agents | Hetero PosDis | HomoVV PosDis |
|----------|--------------|---------------|
| 1 | 0.788 | 0.788 |
| 2 | 0.764 | 0.777 |
| 4 | 0.676 | 0.715 |
| 8 | 0.604 | 0.655 |
| 16 | 0.534 | 0.598 |

Compositionality decreases monotonically with population size but never collapses. Even at 16 agents, PosDis remains above 0.5. The decrease occurs because more agents produce more message positions, making per-position specialization harder.

### Architecture Diversity
Validated with three fundamentally different encoder architectures (Phase 96):
- V-JEPA 2 + DINOv2: PosDis 0.764
- V-JEPA 2 + CLIP: PosDis 0.737
- DINOv2 + CLIP: PosDis 0.657
- V-JEPA 2 + DINOv2 + CLIP (3-architecture pool): PosDis 0.764

Adding a third architecture does not degrade the protocol. Heterogeneous populations achieve +0.044 PosDis over homogeneous populations on average.

## 8. Domain Specificity

Each physics domain requires its own WMCP instance. The protocol captures domain-specific physical relationships that do not transfer across scenarios.

Cross-scenario transfer accuracy (Phase 100, trained on spring):
- Spring → Spring: 78.8% (trained)
- Spring → Fall: 56.5% (near-chance)
- Spring → Ramp: 51.8% (chance)

This means each deployment context (manufacturing inspection, autonomous driving, surgical robotics) requires a separate protocol training. The protocol structure (K, L, architecture) is shared; only the learned projection weights are domain-specific.

## 9. Versioning

WMCP uses semantic versioning: MAJOR.MINOR.PATCH.

- **MAJOR** (breaking): Changes to message format, discretization procedure, or compositionality requirements that break compatibility with existing trained agents.
- **MINOR** (non-breaking): Support for new encoder architectures, additional compositionality metrics, or expanded validation.
- **PATCH**: Bug fixes, documentation updates, clarifications.

A version identifier is included in protocol metadata:

```json
{
  "wmcp_version": "0.1.0",
  "K": 3,
  "L": 2,
  "n_agents": 4,
  "domain": "physics_spring",
  "encoders": ["vjepa2_vitl", "dinov2_vits14"]
}
```

## 10. Known Limitations

1. **Physics properties only.** Validated on mass, elasticity, and friction properties. Performance on semantic, relational, or higher-order properties is unknown.

2. **Co-training required.** Every new encoder requires co-training with existing agents. The protocol is not plug-and-play — zero-shot transfer fails (Phase 97).

3. **No cross-domain transfer.** Spring-trained protocols do not work on fall or ramp scenarios (Phase 100). Each domain needs its own protocol instance.

4. **Maximum validated population: 16 agents.** Larger populations may exhibit further compositionality degradation.

5. **Real-video validation limited.** Validated on Physics 101 dataset (spring, fall, ramp scenarios with 206–1801 clips each). Generalization to in-the-wild video is untested.

6. **Single-property task.** The primary validation task is pairwise mass comparison. Multi-property compositional communication with truly independent attributes has not been validated (Phase 105: synthetic correlated properties produce PosDis = 0 as expected).

7. **Static receiver architecture.** The receiver is retrained from scratch at regular intervals during training. A persistent receiver architecture may improve stability.

8. **No formal convergence guarantees.** Training relies on Gumbel-Softmax relaxation with temperature annealing. Convergence to compositional solutions is empirically robust (1,350 runs, 0 complete failures) but not theoretically guaranteed.
