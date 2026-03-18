# Emergent Compositional Communication for Latent World Properties

**Tomek Kaszyński** · [t.kaszynski@proton.me](mailto:t.kaszynski@proton.me)

> Paper: Forthcoming on arXiv

Neural agents that observe physics videos through different temporal windows develop compositional discrete languages where each symbol position specializes for a distinct physical property — without any supervision on message structure.

## Key Results

| Experiment | Backbone | Agents | Holdout Acc | PosDis | Note |
|-----------|----------|--------|-------------|--------|------|
| Ramp (synthetic) | DINOv2 ViT-S | 4 | 98.3% | 0.999 | 80 seeds, 100% compositional |
| Collision (synthetic) | V-JEPA 2 ViT-L | 4 | 87.4% ± 3.1% | 0.962 | Video-native pretraining |
| Collision (synthetic) | DINOv2 ViT-S | 4 | 77.7% ± 3.9% | 0.904 | Image pretraining |
| Spring mass (real video) | V-JEPA 2 ViT-L | 2 | 84.1% ± 5.6% | — | Physics 101 dataset |

**Scale-matched backbone comparison** (DINOv2 ViT-L vs V-JEPA 2 ViT-L, both 304M params): Cohen's d = 3.37, confirming video-native pretraining drives the advantage over model scale.

**Real-video causal ablation**: Zeroing the mass-relevant agent's message positions drops accuracy by 7.8pp while zeroing the other agent causes only 2.1pp disruption (paired t = 3.63, p = 0.022), confirming compositional encoding on real camera footage.

## Method

1. **Visual encoding**: Frozen V-JEPA 2 or DINOv2 features from physics videos
2. **Multi-agent communication**: Each agent observes a temporal window and sends a discrete message via Gumbel-Softmax
3. **Pairwise comparison task**: A receiver predicts which of two scenes has higher values for each physical property
4. **Iterated learning + population pressure**: Simultaneous receiver reset every 40 epochs with 3 receivers drives compositionality

## Repository Structure

```
├── physics_sim.py          # Physics environments + data generation (Kubric/PyBullet)
├── world_model.py          # Neural networks (slot attention, JEPA, communication)
├── run_all.py              # Experiment launcher functions
├── _phase*.py              # Individual experiment scripts (Phases 44–90)
├── paper_draft.tex         # Paper source
├── EXPERIMENTS.md           # Full experiment log (90 phases)
├── CLAUDE.md               # Project conventions
└── results/
    ├── phase*_*.json       # Experiment results (metrics, per-seed data)
    └── phase*_*.png        # Visualizations
```

### Key experiment scripts

| Script | Description |
|--------|-------------|
| `_phase79_collision_pipeline.py` | Collision dataset: DINOv2 + V-JEPA 2, 2-agent and 4-agent |
| `_phase81_controls.py` | Randomized frames, matched bandwidth, 3-agent controls |
| `_phase82_frame_matched.py` | DINOv2 ViT-L scale-matched backbone control |
| `_phase87_phys101.py` | Physics 101 real-video feature extraction + probes |
| `_phase87b_clean_restitution.py` | Stabilized spring mass training + two-property compositionality |
| `_phase87d_compositionality.py` | Agent scaling sweep on real video (1→2→4 agents) |
| `_phase88_frame_matched.py` | 48-frame DINOv2 frame-count control |
| `_phase89_action_conditioned.py` | Action-conditioned planning with frozen messages |
| `_phase90_realvideo_ablation.py` | Surgical position-zeroing ablation on real video |

## Requirements

```
torch>=2.0
torchvision
numpy
scipy
matplotlib
pillow
pybullet        # for dataset generation
```

For V-JEPA 2 feature extraction:
```
transformers>=4.40
```

Hardware: Tested on M3 MacBook Pro with MPS acceleration (float32 only).

## Quick Start

```bash
# 1. Generate ramp dataset (requires PyBullet)
python3 -c "from physics_sim import generate_ramp_dataset; generate_ramp_dataset()"

# 2. Extract DINOv2 features
python3 -c "from run_all import extract_dino_features; extract_dino_features()"

# 3. Run 2-agent compositional communication (20 seeds)
PYTHONUNBUFFERED=1 python3 -c "from run_all import run_phase54f; run_phase54f()"
```

Results are saved to `results/` as JSON files with per-seed metrics.

## Citation

```bibtex
@article{kaszynski2025emergent,
  title={Emergent Compositional Communication for Latent World Properties},
  author={Kaszy{\'n}ski, Tomek},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT
