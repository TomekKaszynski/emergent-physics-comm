# wmcp — World Model Communication Protocol

Python package for discrete compositional communication between heterogeneous vision foundation models.

## Install

```bash
pip install -e .
```

## Quick Start

```python
from wmcp import Protocol

# Create 2-agent heterogeneous protocol (V-JEPA + DINOv2)
protocol = Protocol(
    agent_configs=[(1024, 4), (384, 4)],  # (input_dim, n_frames) per agent
    vocab_size=3,   # K=3 symbols per position
    n_heads=2,      # L=2 message positions per agent
)

# Encode and communicate
views_a = [vjepa_features_a, dino_features_a]  # scene A
views_b = [vjepa_features_b, dino_features_b]  # scene B
prediction = protocol.communicate(views_a, views_b)
```

## CLI

```bash
wmcp info                                    # Protocol version and config
wmcp benchmark                               # Latency benchmark
wmcp validate --model-path protocol.pt       # Run 9-point compliance suite
wmcp onboard --encoder feat.pt --protocol p.pt --steps 50
```

## Usage Examples

### Validate compositionality

```python
from wmcp.metrics import compute_posdis, compute_topsim, compute_bosdis

posdis, mi_matrix, entropies = compute_posdis(tokens, attributes, vocab_size=3)
topsim = compute_topsim(tokens, mass_bins, obj_bins)
bosdis = compute_bosdis(tokens, attributes, vocab_size=3)
```

### Onboard a new encoder

```python
from wmcp.onboarding import onboard_new_encoder

result = onboard_new_encoder(
    protocol, agent_slot=1,
    new_features=clip_features,
    all_agent_views=updated_views,
    mass_values=mass, obj_names=names,
    new_input_dim=768, n_steps=50)
```

### Run compliance

```python
from wmcp.compliance import validate_protocol

report = validate_protocol(protocol, views, mass, names)
print(f"{report['n_pass']}/{report['n_total']} tests passed")
```

## API Reference

| Module | Function/Class | Description |
|--------|---------------|-------------|
| `wmcp.protocol` | `Protocol` | Multi-agent communication protocol |
| `wmcp.protocol` | `AgentSender` | Single agent: projection + bottleneck |
| `wmcp.protocol` | `Receiver` | Pairwise comparison decoder |
| `wmcp.projection` | `ProjectionLayer` | Trainable encoder projection |
| `wmcp.bottleneck` | `GumbelSoftmaxBottleneck` | Discrete symbol bottleneck |
| `wmcp.metrics` | `compute_posdis()` | Positional Disentanglement |
| `wmcp.metrics` | `compute_topsim()` | Topographic Similarity |
| `wmcp.metrics` | `compute_bosdis()` | Bag-of-Symbols Disentanglement |
| `wmcp.metrics` | `compute_mi_matrix()` | Full MI matrix |
| `wmcp.compliance` | `validate_protocol()` | 9-point compliance suite |
| `wmcp.onboarding` | `onboard_new_encoder()` | Fine-tune new encoder into protocol |
| `wmcp.pubsub` | `MessageBus` | Threaded pub-sub message bus |

## Links

- [Protocol Specification](../protocol-spec/)
- [Paper](https://doi.org/10.5281/zenodo.19197757)

## Tests

```bash
pytest wmcp/tests/ -v
```
