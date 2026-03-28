# Contributing to WMCP

## Quick Start

```bash
git clone https://github.com/TomekKaszynski/emergent-physics-comm.git
cd emergent-physics-comm
pip install -e ".[dev]"
pytest wmcp/tests/ -v
```

## How to Contribute

### Adding a New Encoder

1. Extract frozen features from your encoder (any architecture, any dimension)
2. Create a test script that loads features and runs through the projection layer
3. Run the compliance suite: `python -m wmcp.cli validate --model-path your_protocol.pt`
4. Submit a PR with: feature extraction script, compliance report, model card

See [protocol-spec/ONBOARDING.md](protocol-spec/ONBOARDING.md) for the full onboarding procedure.

### Adding a New Domain Pack

1. Prepare a dataset with video clips and ground-truth physical properties
2. Extract features using at least 2 supported encoders
3. Train the protocol: see `_phase95_realvideo.py` for the training template
4. Validate: PosDis > 0.5, triple metrics reported
5. Submit a PR with: training script, pre-trained weights, dataset card, compliance report

### Bug Reports

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include:
- Python and PyTorch versions
- Exact error message and traceback
- Minimal reproduction script

### Feature Requests

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md). Include:
- Use case description
- Expected behavior
- Whether you're willing to implement it

## Development

### Running Tests

```bash
pytest wmcp/tests/ -v                    # Unit tests
python -m wmcp.benchmarks.benchmark_latency  # Latency benchmark
python -m wmcp.cli info                  # CLI smoke test
```

### Code Style

- Python 3.9+
- Type hints on all public functions
- Docstrings on all public classes and functions
- No dependencies beyond torch, numpy, scipy for core package
- `flush=True` on all print statements in experiment scripts

### PR Guidelines

1. One feature per PR
2. Include tests for new functionality
3. Update relevant documentation
4. Run the full test suite before submitting
5. Reference the relevant experiment phase if applicable

## Architecture

```
wmcp/                  # Core package (pip installable)
├── protocol.py        # Protocol, AgentSender, Receiver
├── projection.py      # ProjectionLayer
├── bottleneck.py      # GumbelSoftmaxBottleneck
├── metrics.py         # PosDis, TopSim, BosDis, MI
├── compliance.py      # 9-point validation suite
├── onboarding.py      # New encoder onboarding
├── monitoring.py      # Drift detection, alerts
├── versioning.py      # Protocol versioning
├── adaptive.py        # Bandwidth adaptation
├── analytics.py       # Export (CSV, JSON, Prometheus)
├── pubsub.py          # Message bus
├── cli.py             # Command-line interface
├── benchmarks/        # Standardized benchmarks
└── tests/             # pytest tests
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
