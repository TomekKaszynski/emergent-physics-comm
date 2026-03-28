# WMCP Examples

## onboard_new_encoder.py

Demonstrates adding a new encoder (CLIP ViT-L/14) to an existing V-JEPA+DINOv2 protocol. Shows convergence in ~50 training steps.

```bash
# Full demo (trains from scratch, ~5 min)
python3 protocol-spec/examples/onboard_new_encoder.py

# Cached results only (instant)
python3 protocol-spec/examples/onboard_new_encoder.py --dry-run
```

## pubsub_demo.py

Two agents communicate through a threaded message bus. Agent A (V-JEPA 2) publishes discrete messages, Agent B (DINOv2) subscribes and decodes.

```bash
python3 protocol-spec/examples/pubsub_demo.py
```

Output shows per-round tokens, predictions, accuracy, and latency.

## Requirements

- PyTorch >= 2.0 with MPS backend (macOS) or CUDA
- Pre-extracted features in `results/` directory (from Phase 87 and Phase 96)
