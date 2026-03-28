"""
WMCP HuggingFace Space — Interactive Demo
==========================================
Gradio app skeleton for a HuggingFace Space.
Upload to: huggingface.co/spaces/wmcp-protocol/wmcp-demo

Run locally: python space_app.py
"""

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("Gradio not installed. Run: pip install gradio")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══ Minimal WMCP architecture (self-contained) ═══

class ProjectionLayer(nn.Module):
    def __init__(self, d, hd=128, nf=4):
        super().__init__()
        ks = min(3, max(1, nf))
        self.t = nn.Sequential(
            nn.Conv1d(d, 256, ks, padding=ks//2), nn.ReLU(),
            nn.Conv1d(256, 128, ks, padding=ks//2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.f = nn.Sequential(nn.Linear(128, hd), nn.ReLU())

    def forward(self, x):
        return self.f(self.t(x.permute(0, 2, 1)).squeeze(-1))


class Agent(nn.Module):
    def __init__(self, d, hd=128, K=3, L=2, nf=4):
        super().__init__()
        self.proj = ProjectionLayer(d, hd, nf)
        self.K = K
        self.heads = nn.ModuleList([nn.Linear(hd, K) for _ in range(L)])

    def forward(self, x):
        h = self.proj(x)
        tokens = [hd(h).argmax(-1) for hd in self.heads]
        return tokens


def run_demo_round(mass_a, mass_b):
    """Simulate a communication round between two agents."""
    # Create synthetic features correlated with mass
    rng = np.random.RandomState(int(mass_a * 100 + mass_b))
    feat_a = rng.randn(1, 4, 1024).astype(np.float32)
    feat_a[0, :, :50] += mass_a * 0.1
    feat_b = rng.randn(1, 4, 1024).astype(np.float32)
    feat_b[0, :, :50] += mass_b * 0.1

    # Generate tokens (untrained — shows message format, not accuracy)
    agent = Agent(1024, 128, 3, 2, 4)
    agent.eval()
    with torch.no_grad():
        tokens_a = [t.item() for t in agent(torch.from_numpy(feat_a))]
        tokens_b = [t.item() for t in agent(torch.from_numpy(feat_b))]

    result = (
        f"Scene A (mass={mass_a:.1f}g): tokens = {tokens_a}\n"
        f"Scene B (mass={mass_b:.1f}g): tokens = {tokens_b}\n\n"
        f"Message format: K=3 symbols × L=2 positions = 9 possible messages\n"
        f"Information capacity: 3.17 bits per agent\n"
        f"Compression: {1024*32}/{2*np.log2(3):.1f} = {1024*32/(2*np.log2(3)):.0f}× vs raw features"
    )
    return result


if HAS_GRADIO:
    demo = gr.Interface(
        fn=run_demo_round,
        inputs=[
            gr.Slider(1, 100, value=25, label="Object A mass (grams)"),
            gr.Slider(1, 100, value=75, label="Object B mass (grams)"),
        ],
        outputs=gr.Textbox(label="Communication Round"),
        title="WMCP — World Model Communication Protocol",
        description=(
            "Discrete compositional communication between heterogeneous vision models.\n\n"
            "This demo shows the message format. For trained protocol results, "
            "see the [Colab notebook](https://github.com/TomekKaszynski/emergent-physics-comm/blob/main/protocol-spec/examples/wmcp_colab_demo.ipynb)."
        ),
        article=(
            "[Paper](https://doi.org/10.5281/zenodo.19197757) · "
            "[Code](https://github.com/TomekKaszynski/emergent-physics-comm) · "
            "[Spec](https://github.com/TomekKaszynski/emergent-physics-comm/tree/main/protocol-spec)"
        ),
    )

    if __name__ == "__main__":
        demo.launch()
else:
    if __name__ == "__main__":
        print(run_demo_round(25.0, 75.0))
