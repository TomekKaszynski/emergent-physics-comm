---
language: en
tags:
- wmcp
- emergent-communication
- world-model
- compositional
- physics
license: apache-2.0
library_name: wmcp
pipeline_tag: feature-extraction
---

# wmcp-{domain}-{scenario}

{description}

## Model Details

- **Protocol version:** WMCP v0.1.0
- **Vocabulary size (K):** {K}
- **Message positions (L):** {L} per agent
- **Physical properties:** {properties}
- **Training data:** {training_data}
- **Validated encoders:** {encoders}

## Performance

| Condition | Accuracy | PosDis | TopSim | BosDis |
|-----------|----------|--------|--------|--------|
| Heterogeneous | {het_acc} | {het_pd} | {het_ts} | {het_bd} |
| Homogeneous V-JEPA | {vv_acc} | {vv_pd} | — | — |
| Homogeneous DINOv2 | {dd_acc} | {dd_pd} | — | — |

## How to Use

```python
from wmcp import Protocol
import torch

protocol = Protocol(
    agent_configs=[(1024, 4), (384, 4)],
    vocab_size={K}, n_heads={L})
protocol.load_state_dict(torch.load("wmcp-{domain}-{scenario}.pt"))

prediction = protocol.communicate(views_a, views_b)
```

## Training Procedure

- **Objective:** Pairwise property comparison (BCE)
- **Optimizer:** Adam (sender lr=1e-3, receiver lr=3e-3)
- **Discretization:** Gumbel-Softmax (τ: 3.0→1.0)
- **Population pressure:** {n_receivers} receivers, reset every {reset_interval} epochs
- **Validation:** Object-level holdout (20%)

## Limitations

{limitations}

## Citation

```bibtex
@article{kaszynski2026emergent,
  title={Emergent Compositional Communication for Latent World Properties},
  author={Kaszyński, Tomek},
  year={2026},
  doi={10.5281/zenodo.19197757}
}
```
