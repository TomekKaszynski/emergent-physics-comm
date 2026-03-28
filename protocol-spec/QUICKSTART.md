# WMCP Quickstart

Get running in under 3 minutes. No GPU required.

## Step 1: Install

```bash
pip install wmcp
# Or from source:
git clone https://github.com/TomekKaszynski/emergent-physics-comm.git
cd emergent-physics-comm
pip install -e .
```

## Step 2: Verify

```bash
python -m wmcp.cli info
```

Expected output:
```
WMCP — World Model Communication Protocol
Version: 0.1.0
Vocabulary (K): 3
Positions (L): 2 per agent
Hidden dim: 128
Validated encoders: V-JEPA 2, DINOv2, CLIP ViT-L/14
```

## Step 3: Benchmark

```bash
python -m wmcp.benchmarks.benchmark_latency
```

Runs 1,000 communication rounds on synthetic data. Expected: ~1ms on CPU.

## Step 4: Run Tests

```bash
pytest wmcp/tests/ -v
```

23 tests, all on synthetic data. No GPU, no downloads.

## Step 5: Full Demo (optional, downloads real models)

```bash
python demo/full_demo.py
```

Downloads DINOv2-L and CLIP-L from HuggingFace (~2GB), extracts features, trains a heterogeneous protocol, runs compliance validation. Takes ~5 minutes on CPU.

## Docker (zero-install)

```bash
docker build -t wmcp .
docker run wmcp info
docker run wmcp benchmark
```

## Next Steps

- [Protocol Specification](SPEC.md) — full technical spec
- [Onboarding Guide](ONBOARDING.md) — add your own encoder
- [API Reference](../wmcp/README.md) — Python API docs
- [Paper](https://doi.org/10.5281/zenodo.19197757) — research paper
