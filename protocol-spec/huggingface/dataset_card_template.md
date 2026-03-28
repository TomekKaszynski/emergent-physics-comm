---
language: en
tags:
- wmcp
- physics
- robotics
- video
license: apache-2.0
task_categories:
- feature-extraction
size_categories:
- 1K<n<10K
---

# wmcp-data-{scenario}

Pre-extracted frozen encoder features for WMCP protocol training on {scenario_name} physics.

## Dataset Description

- **Source:** {source_dataset}
- **Scenarios:** {scenarios}
- **Clips:** {n_clips}
- **Objects:** {n_objects}
- **Properties:** {properties}

## Features

| Encoder | Dimension | Type | File |
|---------|-----------|------|------|
| V-JEPA 2 ViT-L | 1024 | Temporal (8 frames) | `vjepa_features.pt` |
| DINOv2 ViT-S/14 | 384 | Static (middle frame) | `dino_features.pt` |
| CLIP ViT-L/14 | 768 | Static (middle frame) | `clip_features.pt` |

## Usage

```python
import torch

data = torch.load("vjepa_features.pt")
features = data["features"]   # (N, 8, 1024)
obj_names = data["obj_names"]  # List of N strings
mass_values = data["mass_values"]  # (N,) float64
```

## Preprocessing

- V-JEPA 2: 16 frames uniformly sampled, resize 256×256, ImageNet normalization
- DINOv2: Middle frame, resize 224×224, ImageNet normalization, CLS token
- CLIP: Middle frame, CLIP preprocessing (resize 224, center crop, normalize)

## Citation

```bibtex
@article{kaszynski2026emergent,
  title={Emergent Compositional Communication for Latent World Properties},
  author={Kaszyński, Tomek},
  year={2026},
  doi={10.5281/zenodo.19197757}
}
```
