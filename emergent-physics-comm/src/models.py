"""
Neural network architectures for compositional emergent communication.

Sender agents observe temporal sequences of foundation model features and
produce discrete messages via Gumbel-Softmax. Receiver agents decode pairs
of messages to predict pairwise property comparisons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Encode a temporal sequence of foundation model features into a fixed vector.

    Uses 1D convolutions over the time dimension followed by adaptive pooling.

    Args:
        hidden_dim: Output dimension (default: 128).
        input_dim: Per-frame feature dimension (384 for DINOv2 ViT-S, 1024 for ViT-L/V-JEPA 2).
    """

    def __init__(self, hidden_dim=128, input_dim=384):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, n_frames, input_dim) temporal feature sequence.
        Returns:
            (batch, hidden_dim) encoded representation.
        """
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.temporal(x).squeeze(-1)  # (B, 128)
        return self.fc(x)  # (B, hidden_dim)


class CompositionalSender(nn.Module):
    """Sender that produces factored discrete messages via Gumbel-Softmax.

    Each head independently selects a symbol from a vocabulary, producing a
    compositional message structure where positions can specialize for
    different properties.

    Args:
        encoder: TemporalEncoder instance.
        hidden_dim: Encoder output dimension.
        vocab_size: Number of symbols per message position.
        n_heads: Number of message positions (heads).
    """

    def __init__(self, encoder, hidden_dim, vocab_size, n_heads):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)
        ])

    def forward(self, x, tau=1.0, hard=True):
        """
        Args:
            x: (batch, n_frames, input_dim) temporal features.
            tau: Gumbel-Softmax temperature.
            hard: Whether to use straight-through estimator.
        Returns:
            messages: (batch, n_heads * vocab_size) concatenated one-hot vectors.
            all_logits: List of (batch, vocab_size) logits per head.
        """
        h = self.encoder(x)
        messages = []
        all_logits = []
        for head in self.heads:
            logits = head(h)
            if self.training:
                msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
            else:
                idx = logits.argmax(dim=-1)
                msg = F.one_hot(idx, self.vocab_size).float()
            messages.append(msg)
            all_logits.append(logits)
        return torch.cat(messages, dim=-1), all_logits


class MultiAgentSender(nn.Module):
    """Wrapper for multiple CompositionalSenders, one per agent.

    Each agent observes a different temporal window and produces an independent
    message. Messages are concatenated into a joint message.

    Args:
        senders: List of CompositionalSender instances.
    """

    def __init__(self, senders):
        super().__init__()
        self.senders = nn.ModuleList(senders)

    def forward(self, views, tau=1.0, hard=True):
        """
        Args:
            views: List of (batch, frames_per_agent, input_dim) tensors.
            tau: Gumbel-Softmax temperature.
            hard: Whether to use straight-through estimator.
        Returns:
            messages: (batch, n_agents * n_heads * vocab_size) joint message.
            all_logits: List of all head logits across all agents.
        """
        messages = []
        all_logits = []
        for sender, view in zip(self.senders, views):
            msg, logits = sender(view, tau=tau, hard=hard)
            messages.append(msg)
            all_logits.extend(logits)
        return torch.cat(messages, dim=-1), all_logits


class CompositionalReceiver(nn.Module):
    """Receiver that decodes a pair of messages into pairwise property predictions.

    Predicts which of two scenes has a higher value for each physical property
    (binary comparison task).

    Args:
        msg_dim: Total message dimension (n_heads * vocab_size, or n_agents * n_heads * vocab_size).
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.prop1_head = nn.Linear(hidden_dim // 2, 1)
        self.prop2_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, msg_a, msg_b):
        """
        Args:
            msg_a, msg_b: (batch, msg_dim) messages from two scenes.
        Returns:
            pred_prop1, pred_prop2: (batch,) logits for pairwise comparison.
        """
        h = self.shared(torch.cat([msg_a, msg_b], dim=-1))
        return self.prop1_head(h).squeeze(-1), self.prop2_head(h).squeeze(-1)
