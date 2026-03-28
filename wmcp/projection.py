"""Projection layer — the trainable component per encoder."""

import torch
import torch.nn as nn
from typing import Optional


class ProjectionLayer(nn.Module):
    """Learned projection from frozen encoder features to bottleneck input.

    This is the ONLY trainable component per encoder in WMCP.
    Uses 1D temporal convolution followed by adaptive pooling and a linear layer.

    Args:
        input_dim: Feature dimension of the frozen encoder.
        hidden_dim: Output dimension fed to the bottleneck.
        n_frames: Expected number of temporal frames.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 128,
                 n_frames: int = 4):
        super().__init__()
        ks = min(3, max(1, n_frames))
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=ks, padding=ks // 2),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=ks, padding=ks // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project frozen features to bottleneck input.

        Args:
            x: (batch, n_frames, input_dim) temporal feature sequence.

        Returns:
            h: (batch, hidden_dim) projected representation.
        """
        return self.fc(self.temporal(x.permute(0, 2, 1)).squeeze(-1))

    @property
    def n_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
