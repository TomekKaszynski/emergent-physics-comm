# Compositional Emergent Communication for Invisible Physical Properties

Neural agents develop compositional language to communicate invisible physical properties (elasticity, friction, mass ratio) observed through video. Multiple sender agents observe different temporal windows of a physics simulation via frozen foundation model features (DINOv2, V-JEPA 2) and communicate through a discrete Gumbel-Softmax bottleneck. Population-based iterated learning drives the emergence of positionally disentangled protocols where each message position specializes for a distinct physical property. Four-agent systems achieve 98.3% zero-shot generalization to held-out property combinations with near-perfect compositionality (PosDis > 0.99), while matched-bandwidth single-sender controls fail — demonstrating that multi-agent structure, not channel capacity, drives compositional emergence.

**Key result:** At matched model scale (304M parameters), V-JEPA 2 significantly outperforms DINOv2 for physics communication (87.4% vs 74.6%, Cohen's d = 3.37), confirming that video pretraining captures temporal dynamics that image pretraining cannot.

**Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Installation

```bash
git clone https://github.com/tomek-kaszynski/emergent-physics-comm.git
cd emergent-physics-comm
pip install -r requirements.txt
```

## Quick Start

Reproduce the main 4-agent result on the ramp dataset in three commands:

```bash
# 1. Extract DINOv2 features from rendered scenes
python scripts/extract_dinov2.py --dataset data/ramp_dataset --model small --n-frames 8

# 2. Run oracle probe (upper bound)
python scripts/run_oracle.py --features data/dinov2_small_features.pt --n-seeds 20

# 3. Train 4-agent communication with iterated learning
python scripts/train_4agent.py --features data/dinov2_small_features.pt --n-agents 4 --n-seeds 20
```

Pre-extracted features and trained checkpoints will be available on Hugging Face (link TBD).

## Datasets

Two physics environments built with [Kubric](https://github.com/google-research/kubric) + PyBullet:

| Dataset | Scenes | Grid | Properties | Frames |
|---------|--------|------|------------|--------|
| **Ramp** | 300 | 5x5 (elasticity x friction) | Ball slides down 70deg ramp, bounces | 24 @ 12fps |
| **Collision** | 600 | 5x5 (mass ratio x restitution) | Two identical spheres collide | 48 @ 24fps |

Generate datasets (requires Docker + Kubric):
```bash
cd environments
docker run --rm -v "$(pwd):/kubric" kubricdockerhub/kubruntu:latest \
    python3 generate_ramp.py
docker run --rm -v "$(pwd):/kubric" kubricdockerhub/kubruntu:latest \
    python3 generate_collision.py
```

## Project Structure

```
emergent-physics-comm/
├── src/
│   ├── models.py      # Sender, receiver, multi-agent communication
│   ├── training.py    # Iterated learning loop, Gumbel schedule
│   ├── metrics.py     # PosDis, TopSim, mutual information
│   ├── datasets.py    # Feature loading, Latin square holdout
│   └── oracle.py      # Oracle probe (no bottleneck baseline)
├── scripts/
│   ├── train_2agent.py      # 2-agent communication
│   ├── train_4agent.py      # Multi-agent communication
│   ├── run_oracle.py        # Oracle probe training
│   ├── extract_dinov2.py    # DINOv2 feature extraction
│   └── extract_vjepa2.py    # V-JEPA 2 feature extraction
├── environments/
│   ├── generate_ramp.py     # Kubric ramp dataset
│   └── generate_collision.py # Kubric collision dataset
├── configs/
│   └── default.yaml         # Default hyperparameters
└── paper/
    └── paper.tex
```

## Main Results

### Multi-agent scaling (DINOv2, ramp dataset)

| Agents | Holdout Acc | PosDis | Compositional |
|--------|------------|--------|---------------|
| 2 | 76.7% +/- 7.0% | 0.486 | 16/20 |
| 3 | 98.1% +/- 0.9% | 0.998 | 20/20 |
| 4 | 98.3% +/- 1.4% | 0.999 | 80/80 |

### Scale-matched backbone comparison (collision dataset)

| Backbone | Scale | Pretrain | 4-Agent Holdout | PosDis |
|----------|-------|----------|-----------------|--------|
| DINOv2 ViT-S | 22M | Image | 77.7% +/- 3.9% | 0.904 |
| DINOv2 ViT-L | 304M | Image | 74.6% +/- 4.3% | 0.530 |
| V-JEPA 2 ViT-L | 304M | Video | 87.4% +/- 3.1% | 0.962 |

## Citation

```bibtex
@article{kaszynski2026compositional,
  title={Compositional Emergent Communication for Invisible Physical Properties},
  author={Kaszy{\'n}ski, Tomek},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT
