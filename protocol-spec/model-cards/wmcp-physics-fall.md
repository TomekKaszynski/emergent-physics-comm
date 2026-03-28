---
language: en
tags:
- wmcp
- emergent-communication
- world-model
- compositional
- physics
- fall
license: apache-2.0
library_name: wmcp
pipeline_tag: feature-extraction
---

# wmcp-physics-fall

Protocol trained on MIT Physics 101 fall scenario. Objects are dropped from fixed height; agents communicate about mass and bounciness inferred from impact dynamics.

## Model Details

- **Protocol version:** WMCP v0.1.0
- **Vocabulary size (K):** 3 symbols per position
- **Message positions (L):** 2 per agent
- **Message capacity:** 6.3 bits per 2-agent pair
- **Physical properties:** mass, restitution
- **Training data:** MIT Physics 101 — fall scenario (666 clips, 33 objects)
- **Validated encoders:** V-JEPA 2 ViT-L (1024-dim), DINOv2 ViT-S/14 (384-dim), CLIP ViT-L/14 (768-dim)

## Performance

| Condition | Accuracy | PosDis |
|-----------|----------|--------|
| Heterogeneous (V-JEPA+DINOv2) | 86.7% | 0.494 |
| Homogeneous V-JEPA | 85.8% | — |
| Homogeneous DINOv2 | 81.7% | — |

## Intended Use

Quality inspection where material properties must be assessed from observed behavior (e.g., drop tests on production lines).

The protocol enables heterogeneous vision models to communicate about physical scene properties through discrete compositional messages, without requiring shared architecture or explicit representation alignment.

## How to Use

```python
from wmcp import Protocol

# Load protocol (2-agent, K=3)
protocol = Protocol(
    agent_configs=[(1024, 4), (384, 4)],  # V-JEPA + DINOv2
    vocab_size=3, n_heads=2)

# Load weights
protocol.load_state_dict(torch.load("wmcp-physics-fall.pt"))

# Communicate about two scenes
prediction = protocol.communicate(views_a, views_b)
# prediction > 0 means scene A has higher property value
```

## Training Procedure

- **Objective:** Pairwise property comparison via binary cross-entropy
- **Optimizer:** Adam (sender lr=1e-3, receiver lr=3e-3)
- **Discretization:** Gumbel-Softmax (τ: 3.0→1.0, hard after 30 epochs)
- **Population pressure:** 3 receivers, reset every 40 epochs
- **Entropy regularization:** Penalty when position entropy < 0.1
- **Validation:** Object-level holdout (20% of unique objects)

## Limitations

- Validated on fall physics only; does not transfer to other scenarios
- Requires frozen encoder features as input (not raw images)
- Pairwise comparison task only (not absolute property estimation)
- Co-training required for new encoder architectures

## Citation

```bibtex
@article{kaszynski2026emergent,
  title={Emergent Compositional Communication for Latent World Properties},
  author={Kaszy{\'n}ski, Tomek},
  year={2026},
  doi={10.5281/zenodo.19197757}
}
```
