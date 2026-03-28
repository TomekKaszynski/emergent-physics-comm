# Changelog

All notable changes to WMCP are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-28

### Core Protocol
- Discrete compositional communication with K=3 vocabulary, L=2 positions per agent
- Gumbel-Softmax bottleneck with temperature annealing (τ: 3.0→1.0)
- Multi-agent sender/receiver architecture with population pressure
- Validated on 3 encoder architectures: V-JEPA 2, DINOv2, CLIP ViT-L/14
- 1,350+ training runs across 3 physics scenarios (Phase 94)
- Real-video validation on Physics 101 dataset (Phase 95)

### CLI
- `wmcp info` — protocol version and configuration
- `wmcp benchmark` — latency benchmark on synthetic data
- `wmcp validate` — 9-point compliance suite
- `wmcp onboard` — new encoder onboarding
- `wmcp export` / `wmcp load` — portable .wmcp file format
- `wmcp route` — multi-domain protocol routing

### Metrics
- PosDis (Positional Disentanglement)
- TopSim (Topographic Similarity)
- BosDis (Bag-of-Symbols Disentanglement)
- Full MI matrix computation
- Triple metric convergence validated with zero divergences

### Monitoring
- Real-time entropy tracking and drift detection
- Agent enrollment baselines
- JSON Lines log format (Grafana/ELK compatible)
- Prometheus metrics export

### Security
- Threat model documented (6 risks, mitigations for each)
- Adversarial agent detection via entropy anomaly (100% detection rate)
- Agent quarantine system (Phase 125)

### Benchmarks
- Latency: 1.19ms CPU, 7.88ms MPS (Phase 103)
- Throughput: 4,095 comms/s batch mode (Phase 103)
- Compression: 5,200× vs raw features (Phase 113)
- Population scaling: 1–16 agents (Phase 99)
- Onboarding: 50 steps to 90% accuracy (Phase 104)

### API
- FastAPI server with /communicate, /health, /metrics endpoints
- WebSocket real-time message streaming
- Protocol diff tool for version comparison
- Plugin architecture for third-party extensions

### Visualization
- Message frequency heatmaps
- MI(position, attribute) heatmaps
- Onboarding convergence curves
- Prediction scatter plots

### Infrastructure
- pip-installable package (23 unit tests)
- GitHub Actions CI (Python 3.9–3.12)
- Docker container
- ROS 2 integration nodes
- Portable .wmcp file format

### Documentation
- Protocol specification (SPEC.md)
- Architecture diagrams (Mermaid)
- Onboarding guide (7 steps)
- Security assessment
- Competitive comparison
- Expansion roadmap (5 layers)
- HuggingFace model cards
- Contribution guide
