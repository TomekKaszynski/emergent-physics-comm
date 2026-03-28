# WMCP — World Model Communication Protocol

> **v0.1 Draft**

WMCP is a specification for discrete compositional communication between heterogeneous vision foundation models. It enables agents built on different encoder architectures — self-supervised temporal (V-JEPA 2), self-supervised spatial (DINOv2), and language-supervised contrastive (CLIP ViT-L/14) — to develop shared communication protocols about physical scene properties through a learned discrete bottleneck. No alignment maps, no shared architecture, no pre-training coordination. The bottleneck itself is the protocol layer.

## Links

- **Research paper:** [doi:10.5281/zenodo.19197757](https://doi.org/10.5281/zenodo.19197757)
- **Experiment repository:** [github.com/TomekKaszynski/emergent-physics-comm](https://github.com/TomekKaszynski/emergent-physics-comm)

## Documents

| Document | Description |
|----------|-------------|
| [SPEC.md](SPEC.md) | Core protocol specification |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System diagrams (Mermaid) |
| [ONBOARDING.md](ONBOARDING.md) | How to add a new vision encoder |
| [EVIDENCE.md](EVIDENCE.md) | Experimental evidence summary |
| [ROADMAP.md](ROADMAP.md) | Development roadmap |
| [LICENSE](LICENSE) | Apache License 2.0 |

## Status

This specification is derived from 1,350+ training runs across 3 physics scenarios, 3 encoder architectures, 5 codebook sizes, population sizes from 1 to 16 agents, and validated on real camera footage from the Physics 101 dataset. All claims reference specific experimental phases with reproducible metrics.

WMCP v0.1 is a draft specification. It defines the message format, encoding procedure, alignment mechanism, and compositionality requirements based on empirical evidence. It does not yet include a reference implementation or certification tooling.

## License

Apache License 2.0. See [LICENSE](LICENSE).
