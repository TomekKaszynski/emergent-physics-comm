"""Gumbel-Softmax discrete bottleneck module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class GumbelSoftmaxBottleneck(nn.Module):
    """Discrete bottleneck that maps continuous logits to one-hot symbols.

    Uses Gumbel-Softmax relaxation during training for differentiable
    discrete sampling, and deterministic argmax during inference.

    Args:
        vocab_size: Number of symbols per message position (K).
        n_heads: Number of message positions per agent (L).
        hidden_dim: Input dimension from the projection layer.
    """

    def __init__(self, vocab_size: int = 3, n_heads: int = 2,
                 hidden_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)
        ])

    def forward(self, h: torch.Tensor, tau: float = 1.0,
                hard: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Discretize continuous representation into symbols.

        Args:
            h: (batch, hidden_dim) continuous representation.
            tau: Gumbel-Softmax temperature.
            hard: Use straight-through estimator if True.

        Returns:
            message: (batch, n_heads * vocab_size) concatenated one-hot symbols.
            logits: List of (batch, vocab_size) raw logits per position.
        """
        messages: List[torch.Tensor] = []
        all_logits: List[torch.Tensor] = []
        for head in self.heads:
            logits = head(h)
            if self.training:
                msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
            else:
                msg = F.one_hot(
                    logits.argmax(dim=-1), self.vocab_size).float()
            messages.append(msg)
            all_logits.append(logits)
        return torch.cat(messages, dim=-1), all_logits

    def decode_tokens(self, message: torch.Tensor) -> torch.Tensor:
        """Extract integer tokens from a one-hot message.

        Args:
            message: (batch, n_heads * vocab_size) one-hot message.

        Returns:
            tokens: (batch, n_heads) integer token indices.
        """
        reshaped = message.view(-1, self.n_heads, self.vocab_size)
        return reshaped.argmax(dim=-1)
