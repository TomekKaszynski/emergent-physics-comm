"""Core Protocol class — init, encode, decode, communicate."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from wmcp.projection import ProjectionLayer
from wmcp.bottleneck import GumbelSoftmaxBottleneck


class AgentSender(nn.Module):
    """Single agent: projection + bottleneck.

    Args:
        input_dim: Frozen encoder feature dimension.
        hidden_dim: Projection output dimension.
        vocab_size: Symbols per message position (K).
        n_heads: Message positions per agent (L).
        n_frames: Expected temporal frames.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 vocab_size: int = 3, n_heads: int = 2, n_frames: int = 4):
        super().__init__()
        self.projection = ProjectionLayer(input_dim, hidden_dim, n_frames)
        self.bottleneck = GumbelSoftmaxBottleneck(vocab_size, n_heads, hidden_dim)

    def forward(self, x: torch.Tensor, tau: float = 1.0,
                hard: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = self.projection(x)
        return self.bottleneck(h, tau=tau, hard=hard)


class Receiver(nn.Module):
    """Receiver that decodes message pairs into pairwise predictions.

    Args:
        msg_dim: Total message dimension (n_agents * n_heads * vocab_size).
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, msg_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, msg_a: torch.Tensor,
                msg_b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([msg_a, msg_b], dim=-1)).squeeze(-1)


class Protocol(nn.Module):
    """WMCP Protocol: multi-agent compositional communication.

    Manages a population of agent senders and receivers for
    pairwise property comparison through discrete messages.

    Args:
        agent_configs: List of (input_dim, n_frames) per agent.
        hidden_dim: Projection output dimension.
        vocab_size: Symbols per message position (K).
        n_heads: Message positions per agent (L).
        n_receivers: Number of receiver population members.
    """

    def __init__(self, agent_configs: List[Tuple[int, int]],
                 hidden_dim: int = 128, vocab_size: int = 3,
                 n_heads: int = 2, n_receivers: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_agents = len(agent_configs)
        self.msg_dim = self.n_agents * n_heads * vocab_size

        self.senders = nn.ModuleList([
            AgentSender(input_dim, hidden_dim, vocab_size, n_heads, n_frames)
            for input_dim, n_frames in agent_configs
        ])
        self.receivers = nn.ModuleList([
            Receiver(self.msg_dim, hidden_dim)
            for _ in range(n_receivers)
        ])

    def encode(self, views: List[torch.Tensor], tau: float = 1.0,
               hard: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode agent views into a joint discrete message.

        Args:
            views: List of (batch, n_frames, input_dim) per agent.
            tau: Gumbel-Softmax temperature.
            hard: Use straight-through estimator.

        Returns:
            message: (batch, msg_dim) joint message.
            logits: List of per-head logits.
        """
        messages: List[torch.Tensor] = []
        all_logits: List[torch.Tensor] = []
        for sender, view in zip(self.senders, views):
            msg, logits = sender(view, tau=tau, hard=hard)
            messages.append(msg)
            all_logits.extend(logits)
        return torch.cat(messages, dim=-1), all_logits

    def decode(self, msg_a: torch.Tensor,
               msg_b: torch.Tensor) -> torch.Tensor:
        """Decode a message pair using the first receiver.

        Args:
            msg_a: (batch, msg_dim) message from scene A.
            msg_b: (batch, msg_dim) message from scene B.

        Returns:
            prediction: (batch,) comparison logit.
        """
        return self.receivers[0](msg_a, msg_b)

    def communicate(self, views_a: List[torch.Tensor],
                    views_b: List[torch.Tensor],
                    tau: float = 1.0, hard: bool = True
                    ) -> torch.Tensor:
        """Full communication round: encode both scenes, decode comparison.

        Args:
            views_a: List of agent views for scene A.
            views_b: List of agent views for scene B.
            tau: Gumbel-Softmax temperature.
            hard: Straight-through estimator.

        Returns:
            prediction: (batch,) comparison logit.
        """
        msg_a, _ = self.encode(views_a, tau=tau, hard=hard)
        msg_b, _ = self.encode(views_b, tau=tau, hard=hard)
        return self.decode(msg_a, msg_b)

    def extract_tokens(self, views: List[torch.Tensor]) -> torch.Tensor:
        """Extract integer tokens from agent views (inference mode).

        Args:
            views: List of (batch, n_frames, input_dim) per agent.

        Returns:
            tokens: (batch, n_agents * n_heads) integer tokens.
        """
        self.eval()
        with torch.no_grad():
            _, logits = self.encode(views)
            tokens = torch.stack(
                [l.argmax(dim=-1) for l in logits], dim=1)
        return tokens

    def reset_receivers(self) -> None:
        """Reset all receivers to random weights (population pressure)."""
        hidden_dim = self.receivers[0].net[0].in_features // 2
        device = next(self.parameters()).device
        self.receivers = nn.ModuleList([
            Receiver(self.msg_dim, hidden_dim).to(device)
            for _ in range(len(self.receivers))
        ])

    @property
    def info(self) -> Dict:
        """Protocol metadata."""
        return {
            "wmcp_version": "0.1.0",
            "K": self.vocab_size,
            "L": self.n_heads,
            "n_agents": self.n_agents,
            "msg_dim": self.msg_dim,
            "total_params": sum(p.numel() for p in self.parameters()),
            "sender_params": sum(
                p.numel() for s in self.senders for p in s.parameters()),
        }
