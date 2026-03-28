# WMCP Onboarding Guide

How to add a new vision encoder to an existing WMCP deployment.

## Prerequisites

- A frozen vision encoder that produces fixed-dimensional features for video frames or images.
- Access to the target domain's training data (e.g., Physics 101 spring videos for physics properties).
- An existing trained WMCP protocol with at least one operational agent.

## Step 1: Extract Frozen Features

Run your encoder on the training dataset. The encoder must be frozen — no gradients flow through it.

**Input format:** Video clips or images from the target domain.

**Output format:** Feature tensor of shape `(N, T, D)` where:
- `N` = number of clips
- `T` = number of temporal frames (use 1 for static encoders, replicated to match existing agents)
- `D` = encoder feature dimension

Validated dimensions: 384 (DINOv2 ViT-S), 768 (CLIP ViT-L/14), 1024 (V-JEPA 2 ViT-L). Any dimension is supported — the projection layer handles the mapping.

**Normalization:** Use your encoder's standard preprocessing (typically ImageNet mean/std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]).

## Step 2: Initialize Projection Layer

Create a new projection layer for your encoder:

```
Temporal Encoder:
    Conv1d(D, 256, kernel_size=min(3, T), padding=...)
    ReLU
    Conv1d(256, 128, kernel_size=min(3, T), padding=...)
    ReLU
    AdaptiveAvgPool1d(1)
    Linear(128, hidden_dim)
    ReLU

Message Heads (one per position):
    Linear(hidden_dim, K)
```

Default hyperparameters:
- `hidden_dim` = 128
- `K` = 3 (vocabulary size per position)
- `L` = 2 (message positions per agent, matching existing protocol)

Total trainable parameters: approximately 400K. This is intentionally lightweight — the projection layer should be a bottleneck, not a second encoder.

## Step 3: Connect to Bottleneck

Wire the projection output through Gumbel-Softmax discretization:

- **Training:** `F.gumbel_softmax(logits, tau=tau, hard=True)` with temperature annealing from τ = 3.0 to τ = 1.0 over the training period. Use soft (non-hard) Gumbel-Softmax for the first 30 epochs as warmup.
- **Inference:** `F.one_hot(logits.argmax(dim=-1), K).float()` for deterministic symbol selection.

## Step 4: Co-Train with Existing Agents

**Freeze** all existing agent senders and the receiver. **Train** only your new projection layer.

Training setup:
- **Task:** Pairwise mass comparison (or your domain's comparison task).
- **Optimizer:** Adam, learning rate 1e-3.
- **Batch size:** 32.
- **Gradient clipping:** Max norm 1.0.
- **Training signal:** Binary cross-entropy on the existing receiver's predictions.

The new agent's messages pass through the same receiver that the existing agents use. The receiver is frozen — it already understands the protocol's message format. Your agent must learn to produce messages the receiver can interpret.

**Expected convergence:** 50 training steps to reach 90% of the existing protocol's accuracy (Phase 104: 10/10 seeds converged within 50 steps on a Physics 101 spring task with 206 video clips).

## Step 5: Validate Compositionality

After training, verify that your new agent produces compositional messages.

### Required Metrics

1. **PosDis (Positional Disentanglement):** Compute MI between each message position and each physical attribute. PosDis measures the gap between the best and second-best attribute MI per position, normalized by the best.

2. **TopSim (Topographic Similarity):** Sample 5,000 random pairs. Compute Spearman correlation between meaning distances (Manhattan in property space) and message distances (Hamming in symbol space).

3. **BosDis (Bag-of-Symbols Disentanglement):** Like PosDis but computed over symbol identity rather than position — measures whether specific symbols encode specific attributes regardless of which position they appear in.

### Certification Threshold

- **PosDis ≥ 0.5** on held-out objects (20% of unique objects withheld from training).
- TopSim and BosDis reported alongside PosDis. No specific threshold, but values should be proportional to PosDis (Phase 94: all three metrics converge consistently across 1,350 runs with zero divergences).

### Holdout Protocol

Validation must use **object-level** holdout, not sample-level. If an object appears in training, none of its samples may appear in validation. This ensures the protocol generalizes to unseen physical instances.

## Step 6: Generate Compliance Report

Record the following:

```json
{
  "encoder": "your_encoder_name",
  "encoder_dim": 768,
  "wmcp_version": "0.1.0",
  "domain": "physics_spring",
  "K": 3,
  "L": 2,
  "training_steps": 50,
  "holdout_accuracy": 0.828,
  "posdis": 0.737,
  "topsim": 0.437,
  "bosdis": 0.534,
  "n_validation_objects": 5,
  "n_training_objects": 21,
  "certification": "PASS"
}
```

## Step 7: Register Encoder

Add your encoder to the deployment's supported encoders list with:
- Encoder name and version
- Feature dimension
- Training objective (self-supervised, language-supervised, etc.)
- Compliance report from Step 6
- Any preprocessing requirements

## Compute Requirements

Based on Phase 104 benchmarks (Apple M3 Pro, MPS backend):

| Stage | Time | Notes |
|-------|------|-------|
| Feature extraction | 1–5 min | Depends on encoder size and dataset |
| Projection training | < 1 min | 50 steps on 206 clips |
| Validation | < 1 min | Compositionality metrics on full dataset |
| **Total onboarding** | **< 10 min** | Single GPU, no distributed training needed |

The projection layer has approximately 400K parameters. Training requires only forward/backward passes through the projection layer and the frozen receiver — the frozen encoder is called once during feature extraction and cached.
