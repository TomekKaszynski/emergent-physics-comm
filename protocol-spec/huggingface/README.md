# wmcp-protocol

**World Model Communication Protocol** — discrete compositional communication between heterogeneous vision foundation models.

WMCP enables agents built on different encoder architectures (V-JEPA 2, DINOv2, CLIP ViT-L/14) to develop shared communication protocols about physical scene properties through a learned discrete bottleneck. No alignment maps. No shared architecture.

## Models

| Model | Domain | Encoders | Accuracy | PosDis |
|-------|--------|----------|----------|--------|
| [wmcp-physics-spring](wmcp-physics-spring) | Elasticity + friction | V-JEPA 2, DINOv2, CLIP | 81.8% | 0.764 |
| [wmcp-physics-fall](wmcp-physics-fall) | Mass + restitution | V-JEPA 2, DINOv2 | 86.7% | 0.494 |
| [wmcp-physics-ramp](wmcp-physics-ramp) | Surface properties | V-JEPA 2, DINOv2 | 82.1% | 0.520 |

## Links

- [Protocol Specification](https://github.com/TomekKaszynski/emergent-physics-comm/tree/main/protocol-spec)
- [Paper](https://doi.org/10.5281/zenodo.19197757)
- [Code](https://github.com/TomekKaszynski/emergent-physics-comm)
