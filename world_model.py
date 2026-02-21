"""
World Model Architectures — Maps to LeCun's Paper
===================================================
Phase 1: DirectWorldModel = Fig 10(a) — deterministic prediction, no collapse
Phase 3: LatentWorldModel = Fig 12 (JEPA) — encode→predict_latent→decode
Phase 4: VisionWorldModel = JEPA with CNN encoders — learns physics from pixels
Phase 5: MultiAgentWorldModel = Multi-modal JEPA — two views, latent fusion
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================
# PHASE 1: Direct State Prediction (LeCun Fig 10a)
# ============================================================

class DirectWorldModel(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def forward(self, state):
        return self.net(state)

    def rollout(self, initial_state, n_steps):
        import numpy as np
        self.eval()
        states = [initial_state.copy()]
        state = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            for _ in range(n_steps):
                state = self.forward(state)
                states.append(state.squeeze(0).numpy().copy())
        return np.array(states)


# ============================================================
# PHASE 3: Latent Space World Model (LeCun Fig 12 — JEPA)
# ============================================================

class LatentWorldModel(nn.Module):
    """
    LeCun's JEPA at toy scale:
    - Enc(x) → s_x (encoder)
    - Pred(s_x, z) → s̃_y (predictor in latent space)
    - Dec(s̃_y) → ỹ (decoder, for evaluation only)

    Energy = D(s_y, s̃_y) — prediction error in representation space
    """
    def __init__(self, state_dim=4, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim))
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def encode(self, state):
        return self.encoder(state)

    def predict_latent(self, latent):
        return self.predictor(latent)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, state):
        latent = self.encode(state)
        next_latent = self.predict_latent(latent)
        predicted_state = self.decode(next_latent)
        return predicted_state, latent, next_latent

    def rollout(self, initial_state, n_steps):
        import numpy as np
        self.eval()
        states = [initial_state.copy()]
        state_t = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            latent = self.encode(state_t)
            for _ in range(n_steps):
                latent = self.predict_latent(latent)
                states.append(self.decode(latent).squeeze(0).numpy().copy())
        return np.array(states)


# ============================================================
# PHASE 4: Vision World Model (JEPA with CNN — learns from pixels)
# ============================================================

class VisionEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1), nn.ReLU(),   # →16×32×32
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),  # →32×16×16
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # →64×8×8
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU()) # →128×4×4
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x).reshape(x.size(0), -1))


class VisionDecoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid())

    def forward(self, z):
        return self.deconv(self.fc(z).reshape(z.size(0), 128, 4, 4))


class VisionWorldModel(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.encoder = VisionEncoder(latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim))
        self.decoder = VisionDecoder(latent_dim)

    def forward(self, frame):
        latent = self.encoder(frame)
        next_latent = self.predictor(latent)
        return self.decoder(next_latent), latent, next_latent

    def encode(self, frame):
        return self.encoder(frame)

    def predict_latent(self, latent):
        return self.predictor(latent)

    def decode(self, latent):
        return self.decoder(latent)


# ============================================================
# PHASE 5: Multi-Agent Latent Communication
# Two world models talking WITHOUT language — just tensors
# ============================================================

class PartialViewEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU())
        self.fc = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x).reshape(x.size(0), -1))


class MultiAgentWorldModel(nn.Module):
    """
    Agent A sees top-down → encodes to latent_A
    Agent B sees side    → encodes to latent_B
    Fusion: concat → learned MLP → shared latent
    Predictor: shared_latent_t → shared_latent_{t+1}
    Decoder: shared_latent → full state vector

    The fusion module IS the communication channel.
    No words. No tokens. Just tensor exchange.
    """
    def __init__(self, latent_per_agent=16, fused_dim=32, state_dim=4, hidden_dim=128):
        super().__init__()
        self.encoder_a = PartialViewEncoder(latent_per_agent)
        self.encoder_b = PartialViewEncoder(latent_per_agent)
        self.fusion = nn.Sequential(
            nn.Linear(latent_per_agent * 2, fused_dim), nn.ReLU(),
            nn.Linear(fused_dim, fused_dim))
        self.predictor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim))
        self.state_decoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def forward(self, view_a, view_b):
        latent_a = self.encoder_a(view_a)
        latent_b = self.encoder_b(view_b)
        fused = self.fusion(torch.cat([latent_a, latent_b], dim=-1))
        next_fused = self.predictor(fused)
        return self.state_decoder(next_fused), fused

    def encode_agent_a(self, view):
        return self.encoder_a(view)

    def encode_agent_b(self, view):
        return self.encoder_b(view)


class SingleAgentBaseline(nn.Module):
    """Single agent with one view. Baseline to prove fusion helps."""
    def __init__(self, latent_dim=16, state_dim=4, hidden_dim=128):
        super().__init__()
        self.encoder = PartialViewEncoder(latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def forward(self, view):
        return self.predictor(self.encoder(view))


class OcclusionAgentModel(nn.Module):
    """
    Single-agent baseline for occlusion scenario.
    Input: occluded state (some balls zeroed out)
    Output: predicted FULL next state (must guess about unseen balls)
    """
    def __init__(self, state_dim=12, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def forward(self, x):
        return self.net(x)


class OcclusionFusedModel(nn.Module):
    """
    Two agents communicating via latent fusion for occlusion scenario.
    Each agent encodes its occluded view; fusion combines them.
    The fusion module IS the communication channel.
    """
    def __init__(self, state_dim=12, latent_per_agent=24, fused_dim=32, hidden_dim=128):
        super().__init__()
        self.encoder_a = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_per_agent))
        self.encoder_b = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_per_agent))
        self.fusion = nn.Sequential(
            nn.Linear(latent_per_agent * 2, fused_dim), nn.ReLU(),
            nn.Linear(fused_dim, fused_dim))
        self.predictor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim))
        self.state_decoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def forward(self, occ_a, occ_b):
        lat_a = self.encoder_a(occ_a)
        lat_b = self.encoder_b(occ_b)
        fused = self.fusion(torch.cat([lat_a, lat_b], dim=-1))
        next_fused = self.predictor(fused)
        return self.state_decoder(next_fused), fused


class BottleneckedFusionModel(nn.Module):
    """
    OcclusionFusedModel with a variational information bottleneck.
    Each encoder outputs (mu, logvar); z is sampled via reparameterization.
    KL(q(z|x) || N(0,I)) penalizes communication complexity.
    """
    def __init__(self, state_dim=12, comm_dim=24, fused_dim=32, hidden_dim=128):
        super().__init__()
        self.comm_dim = comm_dim
        # Encoders output mu and logvar (2 * comm_dim)
        self.encoder_a = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, comm_dim * 2))
        self.encoder_b = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, comm_dim * 2))
        self.fusion = nn.Sequential(
            nn.Linear(comm_dim * 2, fused_dim), nn.ReLU(),
            nn.Linear(fused_dim, fused_dim))
        self.predictor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim))
        self.state_decoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, occ_a, occ_b):
        ha = self.encoder_a(occ_a)
        mu_a, logvar_a = ha[:, :self.comm_dim], ha[:, self.comm_dim:]
        hb = self.encoder_b(occ_b)
        mu_b, logvar_b = hb[:, :self.comm_dim], hb[:, self.comm_dim:]

        z_a = self.reparameterize(mu_a, logvar_a)
        z_b = self.reparameterize(mu_b, logvar_b)

        fused = self.fusion(torch.cat([z_a, z_b], dim=-1))
        next_fused = self.predictor(fused)
        pred = self.state_decoder(next_fused)

        # KL divergence for both agents
        kl = -0.5 * torch.sum(1 + logvar_a - mu_a.pow(2) - logvar_a.exp(), dim=1)
        kl += -0.5 * torch.sum(1 + logvar_b - mu_b.pow(2) - logvar_b.exp(), dim=1)
        return pred, fused, kl.mean()

    def get_communication_vectors(self, occ_a, occ_b):
        """Return deterministic communication (mu only)."""
        with torch.no_grad():
            ha = self.encoder_a(occ_a)
            hb = self.encoder_b(occ_b)
            return ha[:, :self.comm_dim], hb[:, :self.comm_dim]


# ── Phase 10: Hierarchical Communication ────────────────────────

class HierarchicalFusionModel(nn.Module):
    """Fast (every step) + Slow (every K steps, GRU-based) communication channels."""
    def __init__(self, state_dim=12, fast_dim=4, slow_dim=4,
                 fused_dim=32, hidden_dim=128, slow_update_every=10):
        super().__init__()
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.slow_update_every = slow_update_every
        # Fast encoders (with VIB)
        self.enc_a_fast = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fast_dim * 2))
        self.enc_b_fast = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fast_dim * 2))
        # Slow encoders → GRU input
        self.enc_a_slow_in = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, slow_dim))
        self.enc_b_slow_in = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, slow_dim))
        self.gru_a = nn.GRUCell(slow_dim, slow_dim)
        self.gru_b = nn.GRUCell(slow_dim, slow_dim)
        # Fusion & prediction
        total_comm = 2 * (fast_dim + slow_dim)
        self.fusion = nn.Sequential(
            nn.Linear(total_comm, fused_dim), nn.ReLU(),
            nn.Linear(fused_dim, fused_dim))
        self.predictor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim))
        self.state_decoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward_sequence(self, occ_a_seq, occ_b_seq):
        """Process sequence: (batch, seq_len, state_dim) → predictions, kl_total."""
        B, T, _ = occ_a_seq.shape
        h_a = torch.zeros(B, self.slow_dim, device=occ_a_seq.device)
        h_b = torch.zeros(B, self.slow_dim, device=occ_b_seq.device)
        preds, kl_total = [], 0.0
        fast_vecs_a, fast_vecs_b, slow_vecs_a, slow_vecs_b = [], [], [], []

        for t in range(T):
            oa, ob = occ_a_seq[:, t], occ_b_seq[:, t]
            # Fast channel
            ha_f = self.enc_a_fast(oa)
            mu_af, lv_af = ha_f[:, :self.fast_dim], ha_f[:, self.fast_dim:]
            z_af = self.reparameterize(mu_af, lv_af)
            hb_f = self.enc_b_fast(ob)
            mu_bf, lv_bf = hb_f[:, :self.fast_dim], hb_f[:, self.fast_dim:]
            z_bf = self.reparameterize(mu_bf, lv_bf)
            # Slow channel (update every K steps)
            if t % self.slow_update_every == 0:
                sa_in = self.enc_a_slow_in(oa)
                h_a = self.gru_a(sa_in, h_a)
                sb_in = self.enc_b_slow_in(ob)
                h_b = self.gru_b(sb_in, h_b)
            # Fuse
            fused_in = torch.cat([z_af, z_bf, h_a, h_b], dim=-1)
            fused = self.fusion(fused_in)
            next_fused = self.predictor(fused)
            pred = self.state_decoder(next_fused)
            preds.append(pred)
            fast_vecs_a.append(mu_af.detach())
            fast_vecs_b.append(mu_bf.detach())
            slow_vecs_a.append(h_a.detach())
            slow_vecs_b.append(h_b.detach())
            # KL for fast channel
            kl = -0.5 * torch.sum(1 + lv_af - mu_af.pow(2) - lv_af.exp(), dim=1)
            kl += -0.5 * torch.sum(1 + lv_bf - mu_bf.pow(2) - lv_bf.exp(), dim=1)
            kl_total += kl.mean()

        preds = torch.stack(preds, dim=1)  # (B, T, state_dim)
        kl_total = kl_total / T
        return preds, kl_total, {
            'fast_a': torch.stack(fast_vecs_a, 1),
            'fast_b': torch.stack(fast_vecs_b, 1),
            'slow_a': torch.stack(slow_vecs_a, 1),
            'slow_b': torch.stack(slow_vecs_b, 1),
        }


# ── Phase 11: Vision-Based Occlusion ────────────────────────────

class VisionOcclusionFusedModel(nn.Module):
    """CNN encoders for half-frame images with VIB bottleneck."""
    def __init__(self, state_dim=12, comm_dim=8, fused_dim=32, hidden_dim=128):
        super().__init__()
        self.comm_dim = comm_dim
        def make_cnn():
            return nn.Sequential(
                nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, comm_dim * 2))
        self.encoder_a = make_cnn()
        self.encoder_b = make_cnn()
        self.fusion = nn.Sequential(
            nn.Linear(comm_dim * 2, fused_dim), nn.ReLU(),
            nn.Linear(fused_dim, fused_dim))
        self.predictor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim))
        self.state_decoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, img_a, img_b):
        """img_a, img_b: (B, 3, 64, 64) → pred, fused, kl"""
        ha = self.encoder_a(img_a)
        mu_a, lv_a = ha[:, :self.comm_dim], ha[:, self.comm_dim:]
        hb = self.encoder_b(img_b)
        mu_b, lv_b = hb[:, :self.comm_dim], hb[:, self.comm_dim:]
        z_a = self.reparameterize(mu_a, lv_a)
        z_b = self.reparameterize(mu_b, lv_b)
        fused = self.fusion(torch.cat([z_a, z_b], dim=-1))
        next_fused = self.predictor(fused)
        pred = self.state_decoder(next_fused)
        kl = -0.5 * torch.sum(1 + lv_a - mu_a.pow(2) - lv_a.exp(), dim=1)
        kl += -0.5 * torch.sum(1 + lv_b - mu_b.pow(2) - lv_b.exp(), dim=1)
        return pred, fused, kl.mean()

    def get_communication_vectors(self, img_a, img_b):
        with torch.no_grad():
            ha = self.encoder_a(img_a)
            hb = self.encoder_b(img_b)
            return ha[:, :self.comm_dim], hb[:, :self.comm_dim]


# ── Phase 11b: Vision Autoencoder for pretraining ───────────────

class VisionAutoencoder(nn.Module):
    """CNN autoencoder for half-frame pretraining."""
    def __init__(self, comm_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, comm_dim))
        self.decoder = nn.Sequential(
            nn.Linear(comm_dim, 64), nn.ReLU(),
            nn.Linear(64, 64 * 8 * 8), nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# ── Phase 13: Multi-Agent Fusion ────────────────────────────────

class MultiAgentFusionModel(nn.Module):
    """N agents with bottlenecked encoders, all-to-all fusion."""
    def __init__(self, n_agents, state_dim, comm_dim_per_agent=4,
                 fused_dim=32, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents
        self.cdpa = comm_dim_per_agent
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, comm_dim_per_agent * 2))
            for _ in range(n_agents)])
        total_comm = n_agents * comm_dim_per_agent
        self.fusion = nn.Sequential(
            nn.Linear(total_comm, fused_dim), nn.ReLU(),
            nn.Linear(fused_dim, fused_dim))
        self.predictor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim))
        self.state_decoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, agent_views):
        """agent_views: list of N tensors (B, state_dim)."""
        zs, kl_total = [], 0.0
        for i, enc in enumerate(self.encoders):
            h = enc(agent_views[i])
            mu, lv = h[:, :self.cdpa], h[:, self.cdpa:]
            z = self.reparameterize(mu, lv)
            zs.append(z)
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1)
            kl_total += kl.mean()
        fused = self.fusion(torch.cat(zs, dim=-1))
        pred = self.state_decoder(self.predictor(fused))
        return pred, fused, kl_total / self.n_agents

    def get_communication_vectors(self, agent_views):
        with torch.no_grad():
            mus = []
            for i, enc in enumerate(self.encoders):
                h = enc(agent_views[i])
                mus.append(h[:, :self.cdpa])
            return mus


# ── Phase 17: Scaled Architecture ───────────────────────────────

class ScaledBottleneckedFusionModel(nn.Module):
    """200K+ params vs 23K. Wider, deeper, residual connections, LayerNorm.
    Same 8-dim communication bottleneck."""
    def __init__(self, state_dim=12, comm_dim=8, fused_dim=64, hidden_dim=384):
        super().__init__()
        self.comm_dim = comm_dim
        self.encoder_a = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, comm_dim))
        self.encoder_b = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, comm_dim))
        self.mu_a = nn.Linear(comm_dim, comm_dim)
        self.logvar_a = nn.Linear(comm_dim, comm_dim)
        self.mu_b = nn.Linear(comm_dim, comm_dim)
        self.logvar_b = nn.Linear(comm_dim, comm_dim)
        self.fusion = nn.Sequential(
            nn.Linear(comm_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim))
        self.pred_layer1 = nn.Linear(fused_dim, fused_dim)
        self.pred_norm1 = nn.LayerNorm(fused_dim)
        self.pred_layer2 = nn.Linear(fused_dim, fused_dim)
        self.pred_norm2 = nn.LayerNorm(fused_dim)
        self.decoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))
        self.state_decoder = self.decoder  # alias for planning compat

    def forward(self, occ_a, occ_b):
        h_a, h_b = self.encoder_a(occ_a), self.encoder_b(occ_b)
        mu_a, lv_a = self.mu_a(h_a), self.logvar_a(h_a)
        mu_b, lv_b = self.mu_b(h_b), self.logvar_b(h_b)
        if self.training:
            z_a = mu_a + torch.randn_like(mu_a) * (0.5 * lv_a).exp()
            z_b = mu_b + torch.randn_like(mu_b) * (0.5 * lv_b).exp()
        else:
            z_a, z_b = mu_a, mu_b
        fused = self.fusion(torch.cat([z_a, z_b], dim=-1))
        res = self.pred_norm1(F.relu(self.pred_layer1(fused)))
        res = self.pred_norm2(self.pred_layer2(res))
        fused_next = fused + res
        pred = self.decoder(fused_next)
        kl_a = -0.5 * (1 + lv_a - mu_a.pow(2) - lv_a.exp()).sum(-1).mean()
        kl_b = -0.5 * (1 + lv_b - mu_b.pow(2) - lv_b.exp()).sum(-1).mean()
        mu = torch.cat([mu_a, mu_b], dim=-1)
        return pred, mu, kl_a + kl_b


# ── Phase 14: Planning World Model ──────────────────────────────

class PlanningWorldModel(nn.Module):
    """World model + planning via differentiable MPC.
    Frozen world model (encoder/fusion/predictor/decoder from Phase 6).
    Trainable: goal encoder, policies, action encoder, action predictor.
    """
    def __init__(self, state_dim=12, comm_dim=8, fused_dim=32,
                 action_dim=6, hidden_dim=128, horizon=10):
        super().__init__()
        self.state_dim = state_dim
        self.comm_dim = comm_dim
        self.fused_dim = fused_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.max_force = 2.0

        # World model components (will be loaded & frozen)
        self.encoder_a = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, comm_dim * 2))
        self.encoder_b = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, comm_dim * 2))
        self.fusion = nn.Sequential(
            nn.Linear(comm_dim * 2, fused_dim), nn.ReLU(),
            nn.Linear(fused_dim, fused_dim))
        self.wm_predictor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim))
        self.state_decoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

        # New planning components (trainable)
        self.goal_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim))
        self.policy_a = nn.Sequential(
            nn.Linear(comm_dim + fused_dim + comm_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh())
        self.policy_b = nn.Sequential(
            nn.Linear(comm_dim + fused_dim + comm_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh())
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim))
        self.action_predictor = nn.Sequential(
            nn.Linear(fused_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim))

    def freeze_world_model(self):
        """Freeze encoder/fusion/predictor/decoder (trained from Phase 6)."""
        for mod in [self.encoder_a, self.encoder_b, self.fusion,
                    self.wm_predictor, self.state_decoder]:
            for p in mod.parameters():
                p.requires_grad = False

    def _encode_obs(self, occ_a, occ_b):
        ha = self.encoder_a(occ_a)
        hb = self.encoder_b(occ_b)
        mu_a, mu_b = ha[:, :self.comm_dim], hb[:, :self.comm_dim]
        return mu_a, mu_b

    def imagine_trajectory(self, occ_a, occ_b, goal_state):
        """Imagine H steps. Returns states, actions, comm vectors."""
        B = occ_a.shape[0]
        goal_z = self.goal_encoder(goal_state)
        states = [self.state_decoder(self.fusion(
            torch.cat(self._encode_obs(occ_a, occ_b), dim=-1)))]
        actions_a_list, actions_b_list = [], []
        comm_a_list, comm_b_list = [], []

        cur_oa, cur_ob = occ_a, occ_b
        for t in range(self.horizon):
            mu_a, mu_b = self._encode_obs(cur_oa, cur_ob)
            comm_a_list.append(mu_a)
            comm_b_list.append(mu_b)
            # Policies: own latent + goal + partner message
            act_a = self.policy_a(torch.cat([mu_a, goal_z, mu_b], dim=-1)) * self.max_force
            act_b = self.policy_b(torch.cat([mu_b, goal_z, mu_a], dim=-1)) * self.max_force
            actions_a_list.append(act_a)
            actions_b_list.append(act_b)
            # Fuse observations
            fused = self.fusion(torch.cat([mu_a, mu_b], dim=-1))
            # Encode joint action
            act_z = self.action_encoder(torch.cat([act_a, act_b], dim=-1))
            # Predict next fused state
            next_fused = self.action_predictor(torch.cat([fused, act_z], dim=-1))
            # Decode to state space
            next_state = self.state_decoder(next_fused)
            states.append(next_state)
            # Update observations with occlusion
            cur_oa = next_state.clone()
            cur_ob = next_state.clone()
            for b_idx in range(self.state_dim // 4):
                x = next_state[:, b_idx * 4]
                mask_a = (x >= 1.0).float()
                mask_b = (x < 1.0).float()
                for d in range(4):
                    cur_oa[:, b_idx*4+d] = cur_oa[:, b_idx*4+d] * (1 - mask_a)
                    cur_ob[:, b_idx*4+d] = cur_ob[:, b_idx*4+d] * (1 - mask_b)

        return (torch.stack(states, dim=1),
                torch.stack(actions_a_list, dim=1),
                torch.stack(actions_b_list, dim=1),
                torch.stack(comm_a_list, dim=1),
                torch.stack(comm_b_list, dim=1))

    def plan_loss(self, occ_a, occ_b, goal_state, action_penalty=0.01):
        """Compute planning loss: goal distance + action cost."""
        states, acts_a, acts_b, _, _ = self.imagine_trajectory(occ_a, occ_b, goal_state)
        # Position-only goal distance (ignore velocity in goal)
        pos_idx = []
        for b in range(self.state_dim // 4):
            pos_idx.extend([b*4, b*4+1])
        goal_dist = 0
        for t in range(1, states.shape[1]):
            w = t / states.shape[1]  # weight later steps more
            goal_dist += w * F.mse_loss(states[:, t][:, pos_idx], goal_state[:, pos_idx])
        # Terminal bonus
        goal_dist += 2.0 * F.mse_loss(states[:, -1][:, pos_idx], goal_state[:, pos_idx])
        act_cost = action_penalty * (acts_a.pow(2).mean() + acts_b.pow(2).mean())
        return goal_dist + act_cost, states


# ── Phase 18: Clean Planning Model ──────────────────────────────

class CleanPlanningModel(nn.Module):
    """Planning model with FULLY SEPARATED agent pathways.
    No shared fusion layer — clean 'no-comm' ablation via use_comm flag."""
    def __init__(self, state_dim=12, comm_dim=8, hidden_dim=384,
                 action_dim=6, horizon=15):
        super().__init__()
        self.state_dim = state_dim
        self.comm_dim = comm_dim
        self.horizon = horizon
        self.max_force = 2.0

        # Separate encoders
        self.encoder_a = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, comm_dim))
        self.encoder_b = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, comm_dim))

        # Variational bottleneck
        self.mu_a = nn.Linear(comm_dim, comm_dim)
        self.logvar_a = nn.Linear(comm_dim, comm_dim)
        self.mu_b = nn.Linear(comm_dim, comm_dim)
        self.logvar_b = nn.Linear(comm_dim, comm_dim)

        # Goal encoder
        goal_dim = 32
        self.goal_dim = goal_dim
        self.goal_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim))

        # SEPARATE policies: own_msg + partner_msg + goal → action
        self.policy_a = nn.Sequential(
            nn.Linear(comm_dim + comm_dim + goal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim), nn.Tanh())
        self.policy_b = nn.Sequential(
            nn.Linear(comm_dim + comm_dim + goal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim), nn.Tanh())

        # Predictor: both messages + both actions → next state
        self.predictor = nn.Sequential(
            nn.Linear(comm_dim * 2 + action_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def _encode(self, occ_a, occ_b):
        h_a, h_b = self.encoder_a(occ_a), self.encoder_b(occ_b)
        mu_a, lv_a = self.mu_a(h_a), self.logvar_a(h_a)
        mu_b, lv_b = self.mu_b(h_b), self.logvar_b(h_b)
        if self.training:
            z_a = mu_a + torch.randn_like(mu_a) * (0.5 * lv_a).exp()
            z_b = mu_b + torch.randn_like(mu_b) * (0.5 * lv_b).exp()
        else:
            z_a, z_b = mu_a, mu_b
        kl = -0.5 * (1 + lv_a - mu_a**2 - lv_a.exp()).sum(-1).mean()
        kl += -0.5 * (1 + lv_b - mu_b**2 - lv_b.exp()).sum(-1).mean()
        return z_a, z_b, kl

    def forward(self, occ_a, occ_b, goal, use_comm=True):
        """Single-step forward: encode, communicate, act, predict."""
        z_a, z_b, kl = self._encode(occ_a, occ_b)
        msg_to_a = z_b if use_comm else torch.zeros_like(z_b)
        msg_to_b = z_a if use_comm else torch.zeros_like(z_a)
        g = self.goal_encoder(goal)
        act_a = self.policy_a(torch.cat([z_a, msg_to_a, g], dim=-1)) * self.max_force
        act_b = self.policy_b(torch.cat([z_b, msg_to_b, g], dim=-1)) * self.max_force
        pred = self.predictor(torch.cat([z_a, z_b, act_a, act_b], dim=-1))
        return pred, act_a, act_b, z_a, z_b, kl

    def imagine_trajectory(self, occ_a, occ_b, goal, use_comm=True):
        """Imagine H steps with clean comm knockout."""
        z_a, z_b, kl = self._encode(occ_a, occ_b)
        g = self.goal_encoder(goal)
        states, acts_a, acts_b, comm_a, comm_b = [], [], [], [], []

        for t in range(self.horizon):
            msg_to_a = z_b if use_comm else torch.zeros_like(z_b)
            msg_to_b = z_a if use_comm else torch.zeros_like(z_a)
            act_a = self.policy_a(torch.cat([z_a, msg_to_a, g], dim=-1)) * self.max_force
            act_b = self.policy_b(torch.cat([z_b, msg_to_b, g], dim=-1)) * self.max_force
            pred = self.predictor(torch.cat([z_a, z_b, act_a, act_b], dim=-1))
            states.append(pred)
            acts_a.append(act_a); acts_b.append(act_b)
            comm_a.append(z_a); comm_b.append(z_b)
            # Re-encode from predicted state with occlusion
            cur_oa = pred.clone(); cur_ob = pred.clone()
            n_balls = self.state_dim // 4
            for b_idx in range(n_balls):
                x = pred[:, b_idx * 4]
                mask_a = (x >= 1.0).float()
                mask_b = (x < 1.0).float()
                for d in range(4):
                    cur_oa[:, b_idx*4+d] = cur_oa[:, b_idx*4+d] * (1 - mask_a)
                    cur_ob[:, b_idx*4+d] = cur_ob[:, b_idx*4+d] * (1 - mask_b)
            z_a, z_b, _ = self._encode(cur_oa, cur_ob)

        return (torch.stack(states, dim=1),
                torch.stack(acts_a, dim=1), torch.stack(acts_b, dim=1),
                torch.stack(comm_a, dim=1), torch.stack(comm_b, dim=1))

    def plan_loss(self, occ_a, occ_b, goal, use_comm=True, action_penalty=0.01):
        states, acts_a, acts_b, _, _ = self.imagine_trajectory(occ_a, occ_b, goal, use_comm)
        n_balls = self.state_dim // 4
        pos_idx = []
        for b in range(n_balls):
            pos_idx.extend([b*4, b*4+1])
        goal_dist = 0
        for t in range(states.shape[1]):
            w = (t + 1) / states.shape[1]
            goal_dist += w * F.mse_loss(states[:, t][:, pos_idx], goal[:, pos_idx])
        goal_dist += 2.0 * F.mse_loss(states[:, -1][:, pos_idx], goal[:, pos_idx])
        act_cost = action_penalty * (acts_a.pow(2).mean() + acts_b.pow(2).mean())
        return goal_dist + act_cost, states

    def freeze_world_model(self):
        """Freeze encoders and predictor, keep policies trainable."""
        for m in [self.encoder_a, self.encoder_b, self.mu_a, self.logvar_a,
                  self.mu_b, self.logvar_b, self.predictor]:
            for p in (m.parameters() if hasattr(m, 'parameters') else [m]):
                p.requires_grad = False


# ── Phase 19-20: Affordance Models ──────────────────────────────

class AffordanceGoalEncoder(nn.Module):
    """Encode affordance goals: task_type + subject_state + target_state → 32d."""
    def __init__(self, n_task_types=5, obj_dim=10, goal_dim=32, hidden=128):
        super().__init__()
        self.task_embedding = nn.Embedding(n_task_types, 16)
        self.relationship_encoder = nn.Linear(obj_dim * 2, 8)
        self.encoder = nn.Sequential(
            nn.Linear(16 + obj_dim * 2 + 8, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, goal_dim))

    def forward(self, task_type, subject_state, target_state):
        task_emb = self.task_embedding(task_type)
        rel = self.relationship_encoder(torch.cat([subject_state, target_state], -1))
        return self.encoder(torch.cat([task_emb, subject_state, target_state, rel], -1))


class AffordanceWorldModel(nn.Module):
    """Object-centric world model with affordance bottleneck and self-attention."""
    def __init__(self, obj_dim=10, n_objects=5, affordance_dim=8,
                 comm_dim=8, hidden_dim=256):
        super().__init__()
        self.obj_dim = obj_dim
        self.n_objects = n_objects
        self.affordance_dim = affordance_dim
        enc_dim = hidden_dim // 4

        # Per-object encoder
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, enc_dim))

        # Self-attention for object interactions
        self.interaction_attn = nn.MultiheadAttention(
            embed_dim=enc_dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(enc_dim)

        # Affordance bottleneck per object
        self.affordance_bottleneck = nn.Linear(enc_dim, affordance_dim)

        # Scene encoder
        self.scene_encoder = nn.Sequential(
            nn.Linear(affordance_dim * n_objects, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2))

        # Predictor: scene + actions → next state
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + n_objects * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_objects * obj_dim))

        # Communication encoder
        self.comm_encoder = nn.Linear(hidden_dim // 2, comm_dim)

    def encode_scene(self, objects_state):
        """objects_state: [batch, n_objects, obj_dim] → scene, affordances, attn."""
        obj_repr = self.obj_encoder(objects_state)  # [B, N, enc]
        attn_out, attn_weights = self.interaction_attn(obj_repr, obj_repr, obj_repr)
        obj_repr = self.attn_norm(obj_repr + attn_out)
        affordances = self.affordance_bottleneck(obj_repr)  # [B, N, aff_dim]
        scene = self.scene_encoder(affordances.flatten(-2, -1))  # [B, H/2]
        return scene, affordances, attn_weights

    def forward(self, objects_state, actions=None):
        """Predict next state. objects_state: [B, N, obj_dim], actions: [B, N*2]."""
        scene, affordances, attn_w = self.encode_scene(objects_state)
        if actions is None:
            actions = torch.zeros(objects_state.shape[0], self.n_objects * 2,
                                  device=objects_state.device)
        pred = self.predictor(torch.cat([scene, actions], dim=-1))
        pred = pred.view(-1, self.n_objects, self.obj_dim)
        return pred, affordances, attn_w


# ── Phase 25: Visual World Model ───────────────────────────────

class VisualEncoder(nn.Module):
    """CNN: 64×64×3 → latent_dim. 4 conv layers → flatten → FC."""
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )  # 64→32→16→8→4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class VisualJEPA(nn.Module):
    """Visual JEPA with multi-agent communication bottleneck.
    Two agents encode camera views, exchange 8-dim msgs, predict next embedding.
    Target via EMA encoder. VICReg prevents collapse. ~2.5M params."""

    def __init__(self, latent_dim=256, comm_dim=8, action_dim=4, beta=0.001):
        super().__init__()
        self.latent_dim = latent_dim; self.comm_dim = comm_dim; self.beta = beta
        import copy
        self.encoder = VisualEncoder(latent_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.ema_decay = 0.996
        self.comm_mu_a = nn.Linear(latent_dim, comm_dim)
        self.comm_lv_a = nn.Linear(latent_dim, comm_dim)
        self.comm_mu_b = nn.Linear(latent_dim, comm_dim)
        self.comm_lv_b = nn.Linear(latent_dim, comm_dim)
        pi = latent_dim + comm_dim + action_dim
        self.predictor = nn.Sequential(
            nn.Linear(pi, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def _comm(self, z, mu_l, lv_l):
        mu, lv = mu_l(z), lv_l(z)
        msg = mu + torch.randn_like(mu) * (0.5*lv).exp() if self.training else mu
        kl = -0.5*(1 + lv - mu**2 - lv.exp()).sum(-1).mean()
        return msg, mu, kl

    def _vicreg(self, z, gamma=1.0):
        var_loss = F.relu(gamma - z.std(dim=0)).mean()
        zc = z - z.mean(0)
        cov = (zc.T @ zc) / (z.shape[0]-1)
        off = cov - torch.diag(torch.diag(cov))
        return var_loss + (off**2).mean()

    @torch.no_grad()
    def update_target(self):
        for tp, op in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1-self.ema_decay) * op.data

    def forward(self, img_a, img_b, action, next_a, next_b):
        za = self.encoder(img_a); zb = self.encoder(img_b)
        msg_a, mu_a, kl_a = self._comm(za, self.comm_mu_a, self.comm_lv_a)
        msg_b, mu_b, kl_b = self._comm(zb, self.comm_mu_b, self.comm_lv_b)
        pred_a = self.predictor(torch.cat([za, msg_b, action], -1))
        pred_b = self.predictor(torch.cat([zb, msg_a, action], -1))
        with torch.no_grad():
            tgt_a = self.target_encoder(next_a); tgt_b = self.target_encoder(next_b)
        pred_loss = (F.mse_loss(pred_a, tgt_a) + F.mse_loss(pred_b, tgt_b))/2
        kl = (kl_a + kl_b)/2
        vreg = (self._vicreg(za) + self._vicreg(zb))/2
        total = pred_loss + self.beta * kl + 0.01 * vreg
        return total, pred_loss, kl, vreg, mu_a, mu_b

    def get_messages(self, img_a, img_b):
        za = self.encoder(img_a); zb = self.encoder(img_b)
        return self.comm_mu_a(za), self.comm_mu_b(zb)


# ── Phase 25c: Visual JEPA v2 — Communication Forcing ─────────

class VisualJEPAv2(nn.Module):
    """Visual JEPA with communication forcing mechanisms.

    Changes from VisualJEPA:
    1. β-annealing: beta starts at 0, ramps up during training (set externally)
    2. Comm dropout: randomly zero out agent's OWN latent (forces message use)

    ~3.5M parameters.
    """
    def __init__(self, latent_dim=256, comm_dim=8, action_dim=4,
                 comm_dropout=0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.comm_dim = comm_dim
        self.comm_dropout = comm_dropout
        self.beta = 0.0  # starts at 0, annealed externally during training

        # Encoder (shared between agents)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

        # Target encoder (EMA, no gradients)
        import copy
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.ema_decay = 0.996

        # Communication bottleneck (separate per agent)
        self.comm_mu_a = nn.Linear(latent_dim, comm_dim)
        self.comm_logvar_a = nn.Linear(latent_dim, comm_dim)
        self.comm_mu_b = nn.Linear(latent_dim, comm_dim)
        self.comm_logvar_b = nn.Linear(latent_dim, comm_dim)

        # Predictor: own_latent + partner_message + action → next_latent
        pred_input = latent_dim + comm_dim + action_dim
        self.predictor = nn.Sequential(
            nn.Linear(pred_input, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def _communicate(self, z, mu_layer, logvar_layer):
        mu = mu_layer(z)
        logvar = torch.clamp(logvar_layer(z), -4, 4)
        std = torch.exp(0.5 * logvar)
        msg = mu + std * torch.randn_like(std) if self.training else mu
        kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1).mean()
        return msg, mu, kl

    def _vicreg(self, z, gamma=1.0):
        std = z.std(dim=0)
        var_loss = F.relu(gamma - std).mean()
        z_c = z - z.mean(dim=0)
        cov = (z_c.T @ z_c) / (z.shape[0] - 1)
        off = cov - torch.diag(torch.diag(cov))
        return var_loss + (off ** 2).mean()

    @torch.no_grad()
    def update_target(self):
        for tp, op in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data

    def forward(self, img_a, img_b, action, next_img_a, next_img_b, training=True):
        # Encode
        z_a = self.encoder(img_a)
        z_b = self.encoder(img_b)

        # Communication dropout: randomly zero out own latent
        if training and self.training and self.comm_dropout > 0:
            mask_a = (torch.rand(z_a.shape[0], 1, device=z_a.device) > self.comm_dropout).float()
            mask_b = (torch.rand(z_b.shape[0], 1, device=z_b.device) > self.comm_dropout).float()
            z_a_input = z_a * mask_a
            z_b_input = z_b * mask_b
        else:
            z_a_input = z_a
            z_b_input = z_b

        # Communicate (always from FULL z, not masked)
        msg_a, msg_a_mu, kl_a = self._communicate(z_a, self.comm_mu_a, self.comm_logvar_a)
        msg_b, msg_b_mu, kl_b = self._communicate(z_b, self.comm_mu_b, self.comm_logvar_b)

        # Predict: own (possibly dropped) latent + partner message + action
        z_next_a_pred = self.predictor(torch.cat([z_a_input, msg_b, action], dim=-1))
        z_next_b_pred = self.predictor(torch.cat([z_b_input, msg_a, action], dim=-1))

        # Target
        with torch.no_grad():
            z_next_a_target = self.target_encoder(next_img_a)
            z_next_b_target = self.target_encoder(next_img_b)

        pred_loss = (F.mse_loss(z_next_a_pred, z_next_a_target) +
                     F.mse_loss(z_next_b_pred, z_next_b_target)) / 2
        kl = (kl_a + kl_b) / 2
        vicreg = (self._vicreg(z_a) + self._vicreg(z_b)) / 2

        total = pred_loss + self.beta * kl + 0.01 * vicreg
        return total, pred_loss, kl, vicreg, msg_a_mu, msg_b_mu

    def get_messages(self, img_a, img_b):
        z_a = self.encoder(img_a)
        z_b = self.encoder(img_b)
        return self.comm_mu_a(z_a), self.comm_mu_b(z_b)


# ── Phase 26: Object Discovery via Slot Attention ────────────────

class SlotAttentionModule(nn.Module):
    """Slot Attention: discover objects from spatial features.

    Input: feature map [B, N_positions, D_features]
    Output: slots [B, K, D_slot]

    Slots compete for spatial positions via softmax over the SLOT dimension.
    Reference: Locatello et al. 2020
    """

    def __init__(self, n_slots=6, slot_dim=64, n_iters=3, feature_dim=256):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters

        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
        self.slot_sigma = nn.Parameter(torch.ones(1, 1, slot_dim) * 0.1)

        self.norm_features = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.project_k = nn.Linear(feature_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(feature_dim, slot_dim, bias=False)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2), nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim))
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, features):
        """features: [B, N, D] → slots: [B, K, slot_dim]"""
        B = features.shape[0]
        slots = self.slot_mu + self.slot_sigma * torch.randn(
            B, self.n_slots, self.slot_dim, device=features.device)

        features = self.norm_features(features)
        k = self.project_k(features)
        v = self.project_v(features)

        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)

            scale = self.slot_dim ** -0.5
            attn = torch.bmm(q, k.transpose(1, 2)) * scale
            attn = F.softmax(attn, dim=1)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            updates = torch.bmm(attn, v)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(B, self.n_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class ObjectCentricJEPA(nn.Module):
    """Visual JEPA with object discovery via Slot Attention.

    Architecture:
    1. CNN backbone: 64×64×3 → 8×8×256 feature map
    2. Slot Attention: 64 spatial features → K object slots (slot_dim each)
    3. Communication: compress all slots to comm_dim message through IB
    4. Dynamics: predict next slots from current + action + partner message
    5. Target: EMA encoder produces target slots

    ~3M parameters.
    """

    def __init__(self, n_slots=6, slot_dim=64, comm_dim=8, action_dim=4):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.comm_dim = comm_dim
        self.beta = 0.0  # β-annealing controlled externally

        # CNN backbone (shared)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.GroupNorm(16, 128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.GroupNorm(16, 256), nn.ReLU(),
        )  # 64×64 → 8×8 × 256

        self.pos_embed = nn.Parameter(torch.randn(1, 64, 256) * 0.02)

        self.slot_attention = SlotAttentionModule(
            n_slots=n_slots, slot_dim=slot_dim, n_iters=3, feature_dim=256)

        # Target slot encoder (EMA)
        import copy
        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_pos_embed = nn.Parameter(self.pos_embed.data.clone())
        self.target_pos_embed.requires_grad = False
        self.target_slot_attention = copy.deepcopy(self.slot_attention)
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_slot_attention.parameters():
            p.requires_grad = False
        self.ema_decay = 0.996

        # Communication: compress all slots to comm_dim message
        slot_total = n_slots * slot_dim
        self.comm_enc = nn.Sequential(
            nn.Linear(slot_total, 128), nn.ReLU())
        self.comm_mu = nn.Linear(128, comm_dim)
        self.comm_logvar = nn.Linear(128, comm_dim)

        # Dynamics: predict next slot from current slot + action + message
        self.dynamics = nn.Sequential(
            nn.Linear(slot_dim + action_dim + comm_dim, 128),
            nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, slot_dim))

    def extract_slots(self, image):
        """Image → CNN features → Slot Attention → slots [B, K, slot_dim]."""
        features = self.backbone(image)
        B = features.shape[0]
        features = features.flatten(2).permute(0, 2, 1)  # [B, 64, 256]
        features = features + self.pos_embed
        return self.slot_attention(features)

    def extract_target_slots(self, image):
        """Target encoder (no gradients)."""
        with torch.no_grad():
            features = self.target_backbone(image)
            B = features.shape[0]
            features = features.flatten(2).permute(0, 2, 1)
            features = features + self.target_pos_embed
            return self.target_slot_attention(features)

    def communicate(self, slots):
        """Compress slots to comm_dim message through IB."""
        B = slots.shape[0]
        flat = slots.reshape(B, -1)
        h = self.comm_enc(flat)
        mu = self.comm_mu(h)
        logvar = torch.clamp(self.comm_logvar(h), -4, 4)
        std = torch.exp(0.5 * logvar)
        msg = mu + std * torch.randn_like(std) if self.training else mu
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1).mean()
        return msg, mu, kl

    def predict_next_slots(self, slots, action, partner_msg):
        """Predict next slots from current + action + partner message."""
        B, K, D = slots.shape
        action_exp = action.unsqueeze(1).expand(B, K, -1)
        msg_exp = partner_msg.unsqueeze(1).expand(B, K, -1)
        combined = torch.cat([slots, action_exp, msg_exp], dim=-1)
        return self.dynamics(combined)

    @torch.no_grad()
    def _update_target(self):
        for tp, op in zip(self.target_backbone.parameters(),
                          self.backbone.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data
        for tp, op in zip(self.target_slot_attention.parameters(),
                          self.slot_attention.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data
        self.target_pos_embed.data = (
            self.ema_decay * self.target_pos_embed.data +
            (1 - self.ema_decay) * self.pos_embed.data)

    def _vicreg_slots(self, slots):
        """VICReg on mean slot representation."""
        z = slots.mean(dim=1)
        std = z.std(dim=0)
        var_loss = F.relu(1.0 - std).mean()
        z_c = z - z.mean(dim=0)
        cov = (z_c.T @ z_c) / (z.shape[0] - 1)
        off = cov - torch.diag(torch.diag(cov))
        return var_loss + (off ** 2).mean()

    def _slot_loss(self, pred_slots, target_slots):
        """Greedy matched MSE loss between predicted and target slots."""
        B, K, D = pred_slots.shape
        pred_exp = pred_slots.unsqueeze(2).expand(B, K, K, D)
        tgt_exp = target_slots.unsqueeze(1).expand(B, K, K, D)
        dists = ((pred_exp - tgt_exp) ** 2).sum(-1)  # [B, K, K]

        total_loss = 0
        used = torch.zeros(B, K, device=pred_slots.device, dtype=torch.bool)
        for t in range(K):
            target_col = dists[:, :, t] + used.float() * 1e6
            best_pred = target_col.argmin(dim=1)
            matched = pred_slots[torch.arange(B, device=pred_slots.device), best_pred]
            total_loss = total_loss + F.mse_loss(matched, target_slots[:, t, :])
            used.scatter_(1, best_pred.unsqueeze(1), True)
        return total_loss / K

    def forward(self, img_a, img_b, action, next_img_target_a,
                next_img_target_b, training=True):
        """Forward pass with slot-based encoding and communication."""
        slots_a = self.extract_slots(img_a)
        slots_b = self.extract_slots(img_b)

        msg_a, msg_a_mu, kl_a = self.communicate(slots_a)
        msg_b, msg_b_mu, kl_b = self.communicate(slots_b)

        next_a_pred = self.predict_next_slots(slots_a, action, msg_b)
        next_b_pred = self.predict_next_slots(slots_b, action, msg_a)

        target_slots_a = self.extract_target_slots(next_img_target_a)
        target_slots_b = self.extract_target_slots(next_img_target_b)

        pred_loss = (self._slot_loss(next_a_pred, target_slots_a) +
                     self._slot_loss(next_b_pred, target_slots_b)) / 2
        kl = (kl_a + kl_b) / 2
        vicreg = (self._vicreg_slots(slots_a) + self._vicreg_slots(slots_b)) / 2
        total = pred_loss + self.beta * kl + 0.01 * vicreg

        return total, pred_loss, kl, vicreg, msg_a_mu, msg_b_mu, slots_a, slots_b

    def get_messages(self, img_a, img_b):
        slots_a = self.extract_slots(img_a)
        slots_b = self.extract_slots(img_b)
        _, msg_a_mu, _ = self.communicate(slots_a)
        _, msg_b_mu, _ = self.communicate(slots_b)
        return msg_a_mu, msg_b_mu, slots_a, slots_b


# ── Phase 26b: Object-Centric JEPA v2 (full capacity, no KL) ─────

class ObjectCentricJEPAv2(nn.Module):
    """Object-centric visual JEPA v2.

    Fixes from v1:
    - Full-size backbone (32/64/128/256 channels, >2M params)
    - Deterministic communication (no VAE/KL)
    - Larger dynamics predictor (256-dim hidden)
    """

    def __init__(self, n_slots=6, slot_dim=64, comm_dim=8, action_dim=4):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.comm_dim = comm_dim

        # Full-size CNN backbone (widened for >2M params)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(16, 256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.GroupNorm(16, 512), nn.ReLU(),
        )  # 64→32→16→8, output: [B, 512, 8, 8]

        self.pos_embed = nn.Parameter(torch.randn(1, 64, 512) * 0.02)

        self.slot_attention = SlotAttentionModule(
            n_slots=n_slots, slot_dim=slot_dim, n_iters=3, feature_dim=512)

        # Target encoder (EMA copy)
        import copy
        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_pos_embed = nn.Parameter(self.pos_embed.data.clone())
        self.target_pos_embed.requires_grad = False
        self.target_slot_attention = copy.deepcopy(self.slot_attention)
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_slot_attention.parameters():
            p.requires_grad = False
        self.ema_decay = 0.996

        # Deterministic communication (NO KL, NO VAE)
        slot_total = n_slots * slot_dim  # 384
        self.comm_enc = nn.Sequential(
            nn.Linear(slot_total, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU())
        self.comm_proj = nn.Linear(128, comm_dim)  # → 8-dim message

        # Dynamics predictor (larger, shared across slots)
        self.dynamics = nn.Sequential(
            nn.Linear(slot_dim + action_dim + comm_dim, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, slot_dim))

    def extract_slots(self, image):
        """Image → CNN → Slot Attention → [B, K, slot_dim]."""
        features = self.backbone(image)
        B = features.shape[0]
        features = features.flatten(2).permute(0, 2, 1)  # [B, 64, 256]
        features = features + self.pos_embed
        return self.slot_attention(features)

    def extract_target_slots(self, image):
        """Target encoder (no gradients)."""
        with torch.no_grad():
            features = self.target_backbone(image)
            B = features.shape[0]
            features = features.flatten(2).permute(0, 2, 1)
            features = features + self.target_pos_embed
            return self.target_slot_attention(features)

    def communicate(self, slots):
        """Deterministic 8-dim message. No KL."""
        B = slots.shape[0]
        flat = slots.reshape(B, -1)
        h = self.comm_enc(flat)
        return self.comm_proj(h)

    def predict_next_slots(self, slots, action, partner_msg):
        """Predict next slots from current + action + partner message."""
        B, K, D = slots.shape
        action_exp = action.unsqueeze(1).expand(B, K, -1)
        msg_exp = partner_msg.unsqueeze(1).expand(B, K, -1)
        combined = torch.cat([slots, action_exp, msg_exp], dim=-1)
        return self.dynamics(combined)

    @torch.no_grad()
    def _update_target(self):
        for tp, op in zip(self.target_backbone.parameters(),
                          self.backbone.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data
        for tp, op in zip(self.target_slot_attention.parameters(),
                          self.slot_attention.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data
        self.target_pos_embed.data = (
            self.ema_decay * self.target_pos_embed.data +
            (1 - self.ema_decay) * self.pos_embed.data)

    def _vicreg_slots(self, slots):
        """VICReg on mean slot representation."""
        z = slots.mean(dim=1)
        std = z.std(dim=0)
        var_loss = F.relu(1.0 - std).mean()
        z_c = z - z.mean(dim=0)
        cov = (z_c.T @ z_c) / (z.shape[0] - 1)
        off = cov - torch.diag(torch.diag(cov))
        return var_loss + (off ** 2).mean()

    def _slot_loss(self, pred_slots, target_slots):
        """Greedy matched MSE loss between predicted and target slots."""
        B, K, D = pred_slots.shape
        pred_exp = pred_slots.unsqueeze(2).expand(B, K, K, D)
        tgt_exp = target_slots.unsqueeze(1).expand(B, K, K, D)
        dists = ((pred_exp - tgt_exp) ** 2).sum(-1)

        total_loss = 0
        used = torch.zeros(B, K, device=pred_slots.device, dtype=torch.bool)
        for t in range(K):
            col = dists[:, :, t] + used.float() * 1e6
            best = col.argmin(dim=1)
            matched = pred_slots[
                torch.arange(B, device=pred_slots.device), best]
            total_loss = total_loss + F.mse_loss(matched, target_slots[:, t])
            used.scatter_(1, best.unsqueeze(1), True)
        return total_loss / K

    def forward(self, img_a, img_b, action, next_img_target):
        """Forward pass. No KL — communication is free."""
        slots_a = self.extract_slots(img_a)
        slots_b = self.extract_slots(img_b)

        msg_a = self.communicate(slots_a)
        msg_b = self.communicate(slots_b)

        next_a = self.predict_next_slots(slots_a, action, msg_b)
        next_b = self.predict_next_slots(slots_b, action, msg_a)

        target_slots = self.extract_target_slots(next_img_target)

        pred_loss = (self._slot_loss(next_a, target_slots) +
                     self._slot_loss(next_b, target_slots)) / 2
        vicreg = (self._vicreg_slots(slots_a) +
                  self._vicreg_slots(slots_b)) / 2

        total = pred_loss + 0.01 * vicreg
        return total, pred_loss, vicreg, msg_a, msg_b, slots_a, slots_b

    def get_messages_and_slots(self, img_a, img_b):
        slots_a = self.extract_slots(img_a)
        slots_b = self.extract_slots(img_b)
        msg_a = self.communicate(slots_a)
        msg_b = self.communicate(slots_b)
        return msg_a, msg_b, slots_a, slots_b


# ── Phase 26c: Two-Stage Object Discovery ─────────────────────

class SlotAttentionAutoencoder(nn.Module):
    """Slot Attention autoencoder for object discovery.

    image → CNN → spatial features → slot attention → slots
    slots → spatial broadcast decoder → reconstructed image

    Loss: pixel-level MSE. Proven approach from Locatello 2020.
    Train first, verify slots bind, then use downstream.
    """

    def __init__(self, n_slots=6, slot_dim=64, img_size=64):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.img_size = img_size

        # CNN Encoder: 64×64×3 → 8×8×256
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.GroupNorm(16, 256), nn.ReLU(),
        )  # [B, 256, 8, 8]

        self.pos_embed = nn.Parameter(torch.randn(1, 64, 256) * 0.02)

        self.slot_attention = SlotAttentionModule(
            n_slots=n_slots, slot_dim=slot_dim, n_iters=3, feature_dim=256)

        # Spatial Broadcast Decoder
        # slot_dim + 2 (xy position) → hidden → 4 (RGB + alpha)
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim + 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4),  # 3 RGB + 1 alpha
        )

        # Decoder position grid [-1, 1]
        xs = torch.linspace(-1, 1, img_size)
        ys = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer(
            'grid', torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))

    def encode(self, image):
        """image [B, 3, H, W] → slots [B, K, D]."""
        features = self.encoder(image)
        B = features.shape[0]
        features = features.flatten(2).permute(0, 2, 1)  # [B, 64, 256]
        features = features + self.pos_embed
        return self.slot_attention(features)

    def decode(self, slots):
        """slots [B, K, D] → reconstructed image [B, 3, H, W]."""
        B, K, D = slots.shape
        H = W = self.img_size
        N = H * W

        slots_bc = slots.unsqueeze(2).expand(B, K, N, D)
        grid = self.grid.unsqueeze(0).unsqueeze(0).expand(B, K, N, 2)
        dec_in = torch.cat([slots_bc, grid], dim=-1)  # [B, K, N, D+2]

        decoded = self.decoder(dec_in)  # [B, K, N, 4]
        rgb = decoded[:, :, :, :3]
        alpha = decoded[:, :, :, 3:]

        alpha = F.softmax(alpha, dim=1)  # per-pixel slot assignment
        recon = (alpha * rgb).sum(dim=1)  # [B, N, 3]
        recon = recon.reshape(B, H, W, 3).permute(0, 3, 1, 2)
        return recon, alpha.squeeze(-1)  # alpha: [B, K, N]

    def forward(self, image):
        """Full forward: encode → decode → loss."""
        slots = self.encode(image)
        recon, alpha = self.decode(slots)
        loss = F.mse_loss(recon, image)
        return loss, recon, slots, alpha

    def get_attention_masks(self, image):
        """Per-slot attention masks for visualization."""
        slots = self.encode(image)
        _, alpha = self.decode(slots)
        B, K, N = alpha.shape
        H = W = self.img_size
        return alpha.reshape(B, K, H, W), slots


class SlotAttentionAEv2(nn.Module):
    """Slot Attention AE v2 — fixes slot binding.

    Changes from v1:
    1. Constrained decoder (2 layers, 64 hidden)
    2. Slot dropout during training
    3. Diversity loss pushing slots apart
    """

    def __init__(self, n_slots=6, slot_dim=64, img_size=64):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.img_size = img_size

        # CNN Encoder — EXACTLY [32, 64, 128, 256]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.GroupNorm(16, 256), nn.ReLU(),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 64, 256) * 0.02)

        self.slot_attention = SlotAttentionModule(
            n_slots=n_slots, slot_dim=slot_dim, n_iters=3, feature_dim=256)

        # CONSTRAINED decoder: 2 layers, 64 hidden
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim + 2, 64), nn.ReLU(),
            nn.Linear(64, 4),  # RGB + alpha
        )

        xs = torch.linspace(-1, 1, img_size)
        ys = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer(
            'grid', torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))

    def encode(self, image):
        features = self.encoder(image)
        B = features.shape[0]
        features = features.flatten(2).permute(0, 2, 1)
        features = features + self.pos_embed
        return self.slot_attention(features)

    def decode(self, slots):
        B, K, D = slots.shape
        N = self.img_size * self.img_size

        slots_bc = slots.unsqueeze(2).expand(B, K, N, D)
        grid = self.grid.unsqueeze(0).unsqueeze(0).expand(B, K, N, 2)
        inp = torch.cat([slots_bc, grid], dim=-1)

        decoded = self.decoder(inp)
        rgb = decoded[:, :, :, :3]
        alpha = decoded[:, :, :, 3:]
        alpha = F.softmax(alpha, dim=1)

        recon = (alpha * rgb).sum(dim=1)
        recon = recon.reshape(B, self.img_size, self.img_size, 3).permute(0, 3, 1, 2)
        return recon, alpha.squeeze(-1)

    def diversity_loss(self, slots):
        """Push slots apart via cosine similarity penalty."""
        slots_n = F.normalize(slots, dim=-1)
        sim = torch.bmm(slots_n, slots_n.transpose(1, 2))
        mask = 1 - torch.eye(self.n_slots, device=slots.device).unsqueeze(0)
        return (sim * mask).abs().mean()

    def forward(self, image, training=True):
        import random
        slots = self.encode(image)

        if training:
            B, K, D = slots.shape
            drop_mask = torch.ones(B, K, 1, device=slots.device)
            for b in range(B):
                n_drop = random.randint(1, 2)
                drop_idx = random.sample(range(K), n_drop)
                for d_i in drop_idx:
                    drop_mask[b, d_i, 0] = 0.0
            slots_for_decode = slots * drop_mask
        else:
            slots_for_decode = slots

        recon, alpha = self.decode(slots_for_decode)
        recon_loss = F.mse_loss(recon, image)
        div_loss = self.diversity_loss(slots)
        total = recon_loss + 0.1 * div_loss

        return total, recon_loss, div_loss, recon, slots, alpha

    def get_attention_masks(self, image):
        slots = self.encode(image)
        _, alpha = self.decode(slots)
        B, K, N = alpha.shape
        return alpha.reshape(B, K, self.img_size, self.img_size), slots


class SlotAttentionAEv3(nn.Module):
    """Slot Attention AE v3 — tiny slots for forced distribution.

    slot_dim=12: one slot can hold ~1 object (pos3+color3+size1+shape1≈8).
    10 slots × 12 dims = 120 total for 8 objects × 9 values = 72.
    Standard decoder (not constrained) — bottleneck is slots, not decoder.
    """

    def __init__(self, n_slots=10, slot_dim=12, img_size=64):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.img_size = img_size

        # CNN Encoder — EXACTLY [32, 64, 128, 256]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.GroupNorm(16, 256), nn.ReLU(),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 64, 256) * 0.02)

        self.slot_attention = SlotAttentionModule(
            n_slots=n_slots, slot_dim=slot_dim, n_iters=3, feature_dim=256)

        # Standard spatial broadcast decoder (3 layers, 128 hidden)
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim + 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4),  # RGB + alpha
        )

        xs = torch.linspace(-1, 1, img_size)
        ys = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer(
            'grid', torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))

    def encode(self, image):
        features = self.encoder(image)
        B = features.shape[0]
        features = features.flatten(2).permute(0, 2, 1)
        features = features + self.pos_embed
        return self.slot_attention(features)

    def decode(self, slots):
        B, K, D = slots.shape
        N = self.img_size * self.img_size

        slots_bc = slots.unsqueeze(2).expand(B, K, N, D)
        grid = self.grid.unsqueeze(0).unsqueeze(0).expand(B, K, N, 2)
        inp = torch.cat([slots_bc, grid], dim=-1)

        decoded = self.decoder(inp)
        rgb = decoded[:, :, :, :3]
        alpha = decoded[:, :, :, 3:]
        alpha = F.softmax(alpha, dim=1)

        recon = (alpha * rgb).sum(dim=1)
        recon = recon.reshape(
            B, self.img_size, self.img_size, 3).permute(0, 3, 1, 2)
        return recon, alpha.squeeze(-1)

    def forward(self, image, training=True):
        slots = self.encode(image)
        recon, alpha = self.decode(slots)
        recon_loss = F.mse_loss(recon, image)
        return recon_loss, recon, slots, alpha

    def get_slot_masks(self, image):
        slots = self.encode(image)
        _, alpha = self.decode(slots)
        B, K, N = alpha.shape
        return alpha.reshape(B, K, self.img_size, self.img_size), slots


class SlotJEPAPredictor(nn.Module):
    """Stage 2: JEPA predictor on frozen slot representations.

    Takes pre-extracted slots from the frozen autoencoder.
    Predicts next-step slots using current slots + action + partner message.
    No slot matching needed — frozen encoder gives consistent ordering.
    """

    def __init__(self, n_slots=6, slot_dim=64, comm_dim=8, action_dim=4):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.comm_dim = comm_dim

        # Communication: all slots → 8-dim message (deterministic)
        slot_total = n_slots * slot_dim
        self.comm_proj = nn.Sequential(
            nn.Linear(slot_total, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, comm_dim))

        # Per-slot dynamics predictor
        self.dynamics = nn.Sequential(
            nn.Linear(slot_dim + action_dim + comm_dim, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, slot_dim))

    def communicate(self, slots):
        B = slots.shape[0]
        return self.comm_proj(slots.reshape(B, -1))

    def predict_next(self, slots, action, partner_msg):
        B, K, D = slots.shape
        act_exp = action.unsqueeze(1).expand(B, K, -1)
        msg_exp = partner_msg.unsqueeze(1).expand(B, K, -1)
        inp = torch.cat([slots, act_exp, msg_exp], dim=-1)
        return self.dynamics(inp)

    def forward(self, slots_a, slots_b, action, target_slots):
        """Forward with communication between agents."""
        msg_a = self.communicate(slots_a)
        msg_b = self.communicate(slots_b)

        next_a = self.predict_next(slots_a, action, msg_b)
        next_b = self.predict_next(slots_b, action, msg_a)

        # Simple MSE — no matching needed (frozen encoder, consistent ordering)
        pred_loss = (F.mse_loss(next_a, target_slots) +
                     F.mse_loss(next_b, target_slots)) / 2
        return pred_loss, msg_a, msg_b


# ── Phase 26f: Reference Slot Attention Architecture ──────────

class SoftPositionEmbed(nn.Module):
    """Soft positional embedding from the original Slot Attention paper.

    Creates a 4-channel grid [x, 1-x, y, 1-y] in [0,1] and projects
    through a linear layer, then adds to input features.
    """

    def __init__(self, hidden_dim, grid_size):
        super().__init__()
        H, W = grid_size
        xs = torch.linspace(0, 1, W)
        ys = torch.linspace(0, 1, H)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        pos = torch.stack([xx, 1 - xx, yy, 1 - yy], dim=-1)  # [H, W, 4]
        self.register_buffer('pos', pos)
        self.proj = nn.Linear(4, hidden_dim)

    def forward(self, x):
        # x: [B, H, W, D]
        return x + self.proj(self.pos)


class SlotAttentionModuleV2(nn.Module):
    """Slot Attention with per-slot learnable init (BO-QSA style).

    Key details:
    - Per-slot learnable init: slot_init [1, K, D] — each slot starts from a
      different learned position, breaking symmetry by design (no random sampling)
    - Attention: k @ q^T → [B, N, K], softmax over slots, normalize over inputs
    - 3 iterations (paper default)
    - GRU update + MLP residual per iteration
    """

    def __init__(self, n_slots, slot_dim, n_iters=3, feature_dim=64, epsilon=1e-8):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.epsilon = epsilon

        # Per-slot learnable init (BO-QSA style): each slot starts from a
        # different learned vector, breaking symmetry by design
        self.slot_init = nn.Parameter(torch.randn(1, n_slots, slot_dim) * 0.02)

        # Projection
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(feature_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(feature_dim, slot_dim, bias=False)

        # GRU update
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP residual
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim))

        # Layer norms (on both inputs and slots)
        self.norm_inputs = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs):
        # inputs: [B, N, D_in]
        B, N, _ = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)   # [B, N, D_slot]
        v = self.project_v(inputs)   # [B, N, D_slot]

        # Initialize slots from per-slot learnable vectors (no random sampling)
        slots = self.slot_init.expand(B, -1, -1)

        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)  # [B, K, D]

            # Attention: softmax over SLOTS (competition)
            scale = self.slot_dim ** 0.5
            attn_logits = torch.bmm(k, q.transpose(1, 2)) / scale  # [B, N, K]
            attn = F.softmax(attn_logits, dim=-1) + self.epsilon  # softmax over slots

            # Weighted mean: normalize over inputs (reference order)
            attn = attn / attn.sum(dim=1, keepdim=True)  # [B, N, K]

            updates = torch.bmm(attn.transpose(1, 2), v)  # [B, K, D]

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(B, self.n_slots, self.slot_dim)

            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class SlotAttentionAEv5(nn.Module):
    """Slot Attention Autoencoder matching reference implementation.

    Key properties:
    1. Dense encoder features (64x64 = 4096 tokens, no stride, reference-faithful)
    2. SoftPositionEmbed in encoder
    3. MLP spatial broadcast decoder (pixel-independent, no spatial propagation)
    4. SlotAttentionModuleV2 matching reference: shared init, proper sigma, 3 iters
    5. Pure reconstruction loss (no entropy regularization, matching paper)

    Key insight: CNN decoder causes spatial tiling (Voronoi partition).
    MLP decoder forces object-based decomposition because each pixel is
    decoded independently from slot + position — no spatial info leakage.
    The dense encoder (1024 tokens vs prior 64) gives slot attention enough
    resolution to separate objects.
    """

    def __init__(self, n_slots=10, slot_dim=64, img_size=64):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.img_size = img_size

        # Encoder: NO stride, full resolution (reference-faithful)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=1, padding=2), nn.ReLU(),
        )
        # Encoder init: use PyTorch default (kaiming_uniform_, a=sqrt(5)).
        # This keeps features small so position embedding has relative influence,
        # which helps SA differentiate spatial positions and break symmetry.

        self.encoder_pos = SoftPositionEmbed(64, (64, 64))
        # MLP after position embedding (matching reference architecture:
        # CNN → PosEmbed → MLP → [SA with internal LayerNorm])
        # This MLP mixes content and position info, preventing position dominance
        self.encoder_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.encoder_norm = nn.LayerNorm(64)

        # Slot Attention
        self.slot_attention = SlotAttentionModuleV2(
            n_slots=n_slots, slot_dim=slot_dim, n_iters=3, feature_dim=64)

        # MLP Spatial Broadcast Decoder (pixel-independent)
        # slot_dim + 2 (xy position) → hidden → 4 (RGB + alpha)
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim + 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4),  # 3 RGB + 1 alpha
        )

        # Decoder position grid [-1, 1]
        xs = torch.linspace(-1, 1, img_size)
        ys = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer(
            'grid', torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))

    def encode(self, image):
        x = self.encoder_cnn(image)              # [B, 64, 64, 64] — full resolution
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1)               # [B, 64, 64, 64]
        x = self.encoder_pos(x)                  # add position info
        x = x.reshape(B, 64 * 64, 64)            # [B, 4096, 64]
        x = self.encoder_mlp(x)                  # mix content + position
        x = self.encoder_norm(x)
        return self.slot_attention(x)             # [B, K, slot_dim]

    def decode(self, slots):
        B, K, D = slots.shape
        H = W = self.img_size
        N = H * W

        # Spatial broadcast: each slot to every pixel position
        slots_bc = slots.unsqueeze(2).expand(B, K, N, D)
        grid = self.grid.unsqueeze(0).unsqueeze(0).expand(B, K, N, 2)
        dec_in = torch.cat([slots_bc, grid], dim=-1)  # [B, K, N, D+2]

        decoded = self.decoder(dec_in)  # [B, K, N, 4]
        rgb = decoded[:, :, :, :3]
        alpha_logits = decoded[:, :, :, 3:]

        alpha = F.softmax(alpha_logits, dim=1)  # per-pixel slot competition
        recon = (alpha * rgb).sum(dim=1)  # [B, N, 3]
        recon = recon.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        alpha = alpha.squeeze(-1)  # [B, K, N]
        alpha = alpha.reshape(B, K, H, W)
        return recon, alpha

    def forward(self, image, training=True):
        import numpy as np
        slots = self.encode(image)
        recon, alpha = self.decode(slots)

        recon_loss = F.mse_loss(recon, image)

        # Compute entropy for monitoring only (no regularization — matching reference)
        pixel_entropy = -(alpha * (alpha + 1e-8).log()).sum(dim=1).mean()
        max_entropy = np.log(self.n_slots)
        entropy_reg = pixel_entropy / max_entropy  # normalize to [0, 1]

        total_loss = recon_loss  # pure reconstruction loss, no entropy reg

        return total_loss, recon_loss, entropy_reg, recon, slots, alpha


class SlotAttentionDINO(nn.Module):
    """Slot Attention with frozen DINOv2 encoder and feature reconstruction.

    Key properties:
    1. Frozen DINOv2-Small encoder (dinov2_vits14, ~22M params frozen)
    2. 256 patch tokens (16x16 from 224x224 input), 384-dim features
    3. SlotAttentionModuleV2 with per-slot learnable init (feature_dim=384)
    4. MLP spatial broadcast decoder reconstructs DINOv2 features (not pixels)
    5. Loss: MSE between predicted and actual DINOv2 patch features
    """

    def __init__(self, n_slots=7, slot_dim=64, img_size=64):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.img_size = img_size
        self.dino_dim = 384  # DINOv2-Small output dim
        self.n_patches = 16  # 224 / 14

        # Frozen DINOv2-Small encoder
        self.dino = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        # ImageNet normalization (DINOv2 expects this)
        self.register_buffer('dino_mean',
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('dino_std',
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        # Encoder MLP (process DINOv2 features before SA)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.dino_dim, self.dino_dim),
            nn.ReLU(),
            nn.Linear(self.dino_dim, self.dino_dim),
        )
        self.encoder_norm = nn.LayerNorm(self.dino_dim)

        # Slot Attention (feature_dim=384 for DINOv2, slot_dim=64)
        self.slot_attention = SlotAttentionModuleV2(
            n_slots=n_slots, slot_dim=slot_dim, n_iters=5,
            feature_dim=self.dino_dim)

        # MLP Spatial Broadcast Decoder (reconstructs DINOv2 features)
        # slot_dim + 2 (xy position) → hidden → dino_dim + 1 (features + alpha)
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim + 2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, self.dino_dim + 1),  # 384 features + 1 alpha
        )

        # Decoder position grid for 16x16 patch grid [-1, 1]
        xs = torch.linspace(-1, 1, self.n_patches)
        ys = torch.linspace(-1, 1, self.n_patches)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer(
            'grid', torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))

    def extract_dino_features(self, image):
        """Extract frozen DINOv2 patch features from images.

        Args:
            image: [B, 3, H, W] in [0, 1] range
        Returns:
            patch_tokens: [B, 256, 384] DINOv2 patch features
        """
        with torch.no_grad():
            # Resize to 224x224 (DINOv2 training resolution)
            img_224 = F.interpolate(
                image, size=(224, 224), mode='bilinear', align_corners=False)
            # ImageNet normalization
            img_norm = (img_224 - self.dino_mean) / self.dino_std
            # Extract patch tokens (excluding CLS)
            features = self.dino.forward_features(img_norm)
            return features['x_norm_patchtokens']  # [B, 256, 384]

    def encode(self, image):
        dino_features = self.extract_dino_features(image)  # [B, 256, 384]
        x = self.encoder_mlp(dino_features)
        x = self.encoder_norm(x)
        slots = self.slot_attention(x)  # [B, K, slot_dim]
        return slots, dino_features

    def decode(self, slots):
        B, K, D = slots.shape
        N = self.grid.shape[0]  # 256 patches

        # Spatial broadcast: each slot to every patch position
        slots_bc = slots.unsqueeze(2).expand(B, K, N, D)
        grid = self.grid.unsqueeze(0).unsqueeze(0).expand(B, K, N, 2)
        dec_in = torch.cat([slots_bc, grid], dim=-1)  # [B, K, N, D+2]

        decoded = self.decoder(dec_in)  # [B, K, N, dino_dim+1]
        features = decoded[:, :, :, :self.dino_dim]
        alpha_logits = decoded[:, :, :, self.dino_dim:]

        alpha = F.softmax(alpha_logits, dim=1)  # per-patch slot competition
        recon = (alpha * features).sum(dim=1)  # [B, N, dino_dim]

        alpha = alpha.squeeze(-1)  # [B, K, N]
        return recon, alpha

    def forward(self, image, training=True):
        import numpy as np
        slots, dino_features = self.encode(image)
        recon_features, alpha = self.decode(slots)

        # Loss: MSE on DINOv2 features
        recon_loss = F.mse_loss(recon_features, dino_features)

        # Entropy monitoring (over 256 patches instead of pixels)
        # alpha: [B, K, N] where N=256
        patch_entropy = -(alpha * (alpha + 1e-8).log()).sum(dim=1).mean()
        max_entropy = np.log(self.n_slots)
        entropy_reg = patch_entropy / max_entropy  # normalize to [0, 1]

        total_loss = recon_loss  # pure feature reconstruction

        return total_loss, recon_loss, entropy_reg, recon_features, slots, alpha


class SlotPredictor(nn.Module):
    """Phase 28: Slot dynamics predictor (JEPA in slot space).

    Takes all slots at time t (flattened) and predicts all slots at time t+1.
    Joint prediction allows modeling inter-object interactions.

    Architecture: MLP with LayerNorm, ~226K params.
    Input: [B, n_slots * slot_dim] = [B, 448]
    Output: [B, n_slots * slot_dim] = [B, 448]
    """

    def __init__(self, n_slots=7, slot_dim=64, hidden_dim=256):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        total_dim = n_slots * slot_dim  # 448

        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_dim),
        )

    def forward(self, slots_t):
        """Predict slots at t+1 from slots at t.

        Args:
            slots_t: [B, n_slots, slot_dim]
        Returns:
            slots_pred: [B, n_slots, slot_dim]
        """
        B = slots_t.shape[0]
        flat = slots_t.reshape(B, -1)  # [B, n_slots * slot_dim]
        pred_flat = self.net(flat)      # [B, n_slots * slot_dim]
        return pred_flat.reshape(B, self.n_slots, self.slot_dim)


class MassClassifier(nn.Module):
    """Phase 29b: Classify object mass from centroid trajectory.

    Takes 2D centroid positions for a single object across T frames.
    Computes velocity and acceleration from position deltas.
    Classifies light (m=1) vs heavy (m=3).

    Input: [B, T, 2] — centroid positions over time
    Output: [B, 1] — logit (>0 = heavy)
    """

    def __init__(self, n_frames=20, hidden_dim=64):
        super().__init__()
        # positions T*2 + velocities (T-1)*2 + accelerations (T-2)*2
        input_dim = n_frames * 2 + (n_frames - 1) * 2 + (n_frames - 2) * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, positions):
        """
        Args:
            positions: [B, T, 2] — centroid positions over time
        Returns:
            logit: [B, 1]
        """
        velocities = positions[:, 1:] - positions[:, :-1]       # [B, T-1, 2]
        accelerations = velocities[:, 1:] - velocities[:, :-1]  # [B, T-2, 2]
        features = torch.cat([
            positions.flatten(1),       # [B, T*2]
            velocities.flatten(1),      # [B, (T-1)*2]
            accelerations.flatten(1),   # [B, (T-2)*2]
        ], dim=-1)  # [B, 114]
        return self.net(features)


class MassSender(nn.Module):
    """Phase 30: Sender agent for emergent mass communication.

    Takes raw GT position trajectories for all objects in a scene.
    Must discover which object is heavy and encode it into a discrete message.

    Architecture:
    1. Shared per-object temporal encoder (transformer over time steps)
    2. Cross-object attention (compare dynamics across objects)
    3. Gumbel-softmax message head (discrete bottleneck)

    Input: [B, n_obj, T, 2] — position trajectories
    Output: message_onehot [B, vocab_size] (soft during training, hard at eval)
    """

    def __init__(self, n_frames=40, d_model=32, nhead=2, n_layers=2,
                 vocab_size=8, n_obj=3):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_obj = n_obj

        # Per-object temporal encoder (shared across objects)
        self.input_proj = nn.Linear(2, d_model)

        # Sinusoidal positional encoding (fixed)
        pe = torch.zeros(n_frames, d_model)
        pos = torch.arange(0, n_frames, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, T, d]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=0.0, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # Cross-object attention
        cross_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=0.0, batch_first=True)
        self.cross_obj_attn = nn.TransformerEncoder(
            cross_layer, num_layers=1)

        # Message head
        self.message_head = nn.Linear(d_model, vocab_size)

    def forward(self, trajectories, tau=1.0, hard=False, continuous=False):
        """
        Args:
            trajectories: [B, n_obj, T, 2]
            tau: Gumbel-softmax temperature
            hard: if True, use argmax (eval mode)
            continuous: if True, return raw logits (no discretization)
        Returns:
            message: [B, vocab_size] — continuous, soft one-hot, or hard one-hot
        """
        B, K, T, _ = trajectories.shape

        # 1. Per-object temporal encoding (shared weights)
        x = trajectories.reshape(B * K, T, 2)       # [B*K, T, 2]
        x = self.input_proj(x)                       # [B*K, T, d]
        x = x + self.pe[:, :T, :]                    # add positional encoding
        x = self.temporal_encoder(x)                  # [B*K, T, d]
        x = x.mean(dim=1)                            # [B*K, d] mean pool over time
        x = x.reshape(B, K, self.d_model)            # [B, K, d]

        # 2. Cross-object attention
        x = self.cross_obj_attn(x)                   # [B, K, d]
        x = x.mean(dim=1)                            # [B, d] pool over objects

        # 3. Message output
        logits = self.message_head(x)                # [B, vocab_size]
        if continuous:
            return logits                            # raw 8-dim vector
        if hard:
            idx = logits.argmax(dim=-1)
            message = F.one_hot(idx, self.vocab_size).float()
        else:
            message = F.gumbel_softmax(logits, tau=tau, hard=True)
        return message


class MassReceiver(nn.Module):
    """Phase 30: Receiver agent for emergent mass communication.

    Takes a discrete message and predicts which object index is heavy.

    Input: message [B, vocab_size] one-hot
    Output: logits [B, n_obj] — which object is heavy
    """

    def __init__(self, vocab_size=8, d_model=32, n_obj=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_obj),
        )

    def forward(self, message):
        return self.net(message)


def count_params(model):
    return sum(p.numel() for p in model.parameters())

def print_model_info(name, model):
    n = count_params(model)
    print(f"{name}: {n/1e3:.1f}K params" if n < 1e6 else f"{name}: {n/1e6:.1f}M params")


if __name__ == "__main__":
    print("World Model Architectures")
    print("=" * 50)
    for name, m in [
        ("Phase 1 — Direct MLP", DirectWorldModel()),
        ("Phase 3 — Latent JEPA", LatentWorldModel()),
        ("Phase 4 — Vision", VisionWorldModel()),
        ("Phase 5 — Multi-Agent", MultiAgentWorldModel()),
        ("Phase 5 — Single Baseline", SingleAgentBaseline()),
    ]:
        print_model_info(name, m)

    # Quick forward pass tests
    x = torch.randn(8, 4)
    print(f"\nDirect: {x.shape} → {DirectWorldModel()(x).shape}")
    pred, lat, nlat = LatentWorldModel()(x)
    print(f"Latent: {x.shape} → latent {lat.shape} → pred {pred.shape}")
    f = torch.randn(8, 3, 64, 64)
    pf, lat, _ = VisionWorldModel()(f)
    print(f"Vision: {f.shape} → latent {lat.shape} → pred {pf.shape}")
    ps, fused = MultiAgentWorldModel()(f, f)
    print(f"Multi-Agent: 2×{f.shape} → fused {fused.shape} → pred {ps.shape}")

