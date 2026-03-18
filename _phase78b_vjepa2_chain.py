"""
Phase 78-probe + 78b + 78c: V-JEPA 2 proper aggregation chain
==============================================================
Three experiments in one script — do NOT stop between them.

Phase 78-probe: Attentive probe (4 learned queries cross-attending to 2048 tokens)
  5 seeds, 100 epochs. Diagnostic only.

Phase 78b: 2-agent Perceiver resampler with factorized hierarchical attention
  Per-frame spatial cross-attention (8 queries × 256 tokens) → temporal cross-attention
  (4 queries × 64 tokens). 20 seeds, 400 epochs.

Phase 78c: 4-agent temporal split with per-agent Perceiver
  Each agent gets 2 temporal positions (512 tokens), Perceiver K=4 queries, 2 blocks.
  20 seeds, 400 epochs.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase78b_vjepa2_chain.py
"""

import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# Shared configuration
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
VJEPA_DIM = 1024
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

ORACLE_EPOCHS = 100
ORACLE_LR = 1e-3
COMM_EPOCHS = 400
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3


# ══════════════════════════════════════════════════════════════════
# Shared architecture components
# ══════════════════════════════════════════════════════════════════

class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to key-value pairs."""
    def __init__(self, query_dim, kv_dim, n_heads=4, head_dim=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner_dim = n_heads * head_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)

    def forward(self, queries, kv):
        """queries: (B, Nq, D_q), kv: (B, Nkv, D_kv) -> (B, Nq, D_q)"""
        B, Nq, _ = queries.shape
        Nkv = kv.shape[1]
        q = self.to_q(self.norm_q(queries))
        k = self.to_k(self.norm_kv(kv))
        v = self.to_v(kv)

        q = q.view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Nkv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Nkv, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.to_out(out), attn


class CompositionalSender(nn.Module):
    def __init__(self, encoder, hidden_dim, vocab_size, n_heads):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)
        ])

    def forward(self, x, tau=1.0, hard=True):
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
    """Wraps N individual CompositionalSenders."""
    def __init__(self, senders):
        super().__init__()
        self.senders = nn.ModuleList(senders)

    def forward(self, views, tau=1.0, hard=True):
        messages = []
        all_logits = []
        for sender, view in zip(self.senders, views):
            msg, logits = sender(view, tau=tau, hard=hard)
            messages.append(msg)
            all_logits.extend(logits)
        return torch.cat(messages, dim=-1), all_logits


class CompositionalReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim
        self.shared = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.elast_head = nn.Linear(hidden_dim // 2, 1)
        self.friction_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, msg_a, msg_b):
        h = self.shared(torch.cat([msg_a, msg_b], dim=-1))
        return self.elast_head(h).squeeze(-1), self.friction_head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# Phase 78-probe: Attentive probe encoder
# ══════════════════════════════════════════════════════════════════

class AttentiveProbeEncoder(nn.Module):
    """4 learned queries cross-attend to 2048 V-JEPA 2 tokens → 128-dim."""
    def __init__(self, hidden_dim=128, input_dim=1024, n_queries=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, input_dim) * 0.02)
        self.cross_attn = CrossAttention(input_dim, input_dim, n_heads=4, head_dim=64)
        self.proj = nn.Sequential(
            nn.Linear(n_queries * input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, 2048, 1024)
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        out, _ = self.cross_attn(q, x)  # (B, 4, 1024)
        return self.proj(out.reshape(B, -1))  # (B, 128)


# ══════════════════════════════════════════════════════════════════
# Phase 78b: Factorized hierarchical Perceiver encoder
# ══════════════════════════════════════════════════════════════════

class FactorizedPerceiverEncoder(nn.Module):
    """
    Factorized hierarchical attention for V-JEPA 2 tokens (8×256×1024).
    (1) Per-frame spatial: 8 learned queries cross-attend to 256 spatial tokens
        Output: (B, 8, 8, 1024) → reshape (B, 64, 1024)
    (2) Global temporal: 4 learned queries cross-attend to 64 compressed tokens
        Output: (B, 4, 1024) → flatten → MLP → 128-dim
    """
    def __init__(self, hidden_dim=128, input_dim=1024,
                 spatial_queries=8, temporal_queries=4):
        super().__init__()
        self.spatial_queries = nn.Parameter(
            torch.randn(1, spatial_queries, input_dim) * 0.02)
        self.spatial_attn = CrossAttention(
            input_dim, input_dim, n_heads=4, head_dim=64)

        n_spatial_out = spatial_queries * 8  # 64
        self.temporal_queries = nn.Parameter(
            torch.randn(1, temporal_queries, input_dim) * 0.02)
        self.temporal_attn = CrossAttention(
            input_dim, input_dim, n_heads=4, head_dim=64)

        self.proj = nn.Sequential(
            nn.Linear(temporal_queries * input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, 2048, 1024) → reshape to (B, 8, 256, 1024)
        B = x.shape[0]
        x = x.view(B, 8, 256, -1)

        # (1) Per-frame spatial cross-attention
        spatial_out = []
        sq = self.spatial_queries.expand(B, -1, -1)  # (B, 8, 1024)
        for t in range(8):
            frame_tokens = x[:, t, :, :]  # (B, 256, 1024)
            out_t, _ = self.spatial_attn(sq, frame_tokens)  # (B, 8, 1024)
            spatial_out.append(out_t)
        spatial_out = torch.cat(spatial_out, dim=1)  # (B, 64, 1024)

        # (2) Global temporal cross-attention
        tq = self.temporal_queries.expand(B, -1, -1)  # (B, 4, 1024)
        temporal_out, _ = self.temporal_attn(tq, spatial_out)  # (B, 4, 1024)

        return self.proj(temporal_out.reshape(B, -1))  # (B, 128)


# ══════════════════════════════════════════════════════════════════
# Phase 78c: Per-agent Perceiver encoder (for temporal split)
# ══════════════════════════════════════════════════════════════════

class AgentPerceiverEncoder(nn.Module):
    """
    Perceiver resampler for one agent's temporal slice.
    K=4 learned queries, 2 cross-attention blocks.
    Input: (B, 512, 1024) — 2 temporal positions × 256 spatial tokens.
    Output: (B, 128)
    """
    def __init__(self, hidden_dim=128, input_dim=1024, n_queries=4, n_blocks=2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, input_dim) * 0.02)
        self.blocks = nn.ModuleList([
            CrossAttention(input_dim, input_dim, n_heads=4, head_dim=64)
            for _ in range(n_blocks)
        ])
        self.proj = nn.Sequential(
            nn.Linear(n_queries * input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, 512, 1024)
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        for block in self.blocks:
            out, _ = block(q, x)
            q = q + out  # residual
        return self.proj(q.reshape(B, -1))


# ══════════════════════════════════════════════════════════════════
# Oracle variants
# ══════════════════════════════════════════════════════════════════

class Oracle(nn.Module):
    """Single-encoder oracle for 2-agent setups."""
    def __init__(self, encoder_cls, encoder_kwargs, hidden_dim):
        super().__init__()
        self.enc_a = encoder_cls(**encoder_kwargs)
        self.enc_b = encoder_cls(**encoder_kwargs)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.elast_head = nn.Linear(hidden_dim, 1)
        self.friction_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_a, x_b):
        ha = self.enc_a(x_a)
        hb = self.enc_b(x_b)
        h = self.shared(torch.cat([ha, hb], dim=-1))
        return self.elast_head(h).squeeze(-1), self.friction_head(h).squeeze(-1)


class MultiAgentOracle(nn.Module):
    """Oracle with N agent encoders for 4-agent setup."""
    def __init__(self, n_agents, encoder_cls, encoder_kwargs, hidden_dim):
        super().__init__()
        self.n_agents = n_agents
        self.encs_a = nn.ModuleList([
            encoder_cls(**encoder_kwargs) for _ in range(n_agents)
        ])
        self.encs_b = nn.ModuleList([
            encoder_cls(**encoder_kwargs) for _ in range(n_agents)
        ])
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * n_agents * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.elast_head = nn.Linear(hidden_dim, 1)
        self.friction_head = nn.Linear(hidden_dim, 1)

    def forward(self, views_a, views_b):
        ha = torch.cat([enc(v) for enc, v in zip(self.encs_a, views_a)], dim=-1)
        hb = torch.cat([enc(v) for enc, v in zip(self.encs_b, views_b)], dim=-1)
        h = self.shared(torch.cat([ha, hb], dim=-1))
        return self.elast_head(h).squeeze(-1), self.friction_head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════

def load_vjepa_full_features(path):
    """Load full V-JEPA 2 token grid: (300, 2048, 1024) float16 → float32."""
    data = torch.load(path, weights_only=False)
    features = data['features']  # keep as float16 to save memory
    index = data['index']
    e_bins = np.array([entry['elasticity_bin'] for entry in index])
    f_bins = np.array([entry['friction_bin'] for entry in index])
    return features, e_bins, f_bins


def create_splits(e_bins, f_bins, holdout_cells):
    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in holdout_cells:
            holdout_ids.append(i)
        else:
            train_ids.append(i)
    return np.array(train_ids), np.array(holdout_ids)


def sample_pairs(scene_ids, batch_size, rng):
    idx_a = rng.choice(scene_ids, size=batch_size)
    idx_b = rng.choice(scene_ids, size=batch_size)
    same = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.choice(scene_ids, size=same.sum())
        same = idx_a == idx_b
    return idx_a, idx_b


def split_temporal_views(data_f16, indices):
    """Split (N, 2048, 1024) into 4 agent views of (N, 512, 1024).
    Reshape: (N, 2048, 1024) → (N, 8, 256, 1024) → 4 × (N, 2, 256, 1024) → (N, 512, 1024)
    """
    # Select and cast to float32
    selected = data_f16[indices].float()  # (B, 2048, 1024)
    B = selected.shape[0]
    reshaped = selected.view(B, 8, 256, -1)  # (B, 8, 256, 1024)
    views = []
    for i in range(4):
        t_start = i * 2
        agent_view = reshaped[:, t_start:t_start+2, :, :].reshape(B, 512, -1)
        views.append(agent_view)
    return views


# ══════════════════════════════════════════════════════════════════
# Evaluation (2-agent flat)
# ══════════════════════════════════════════════════════════════════

def evaluate_accuracy_flat(sender, receiver, data_f16, e_bins, f_bins,
                           scene_ids, device, oracle_model=None, n_rounds=30):
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    correct_e = correct_f = correct_both = 0
    total_e = total_f = total_both = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)
        da = data_f16[ia].float().to(device)
        db = data_f16[ib].float().to(device)

        label_e = (e_dev[ia] > e_dev[ib])
        label_f = (f_dev[ia] > f_dev[ib])

        if oracle_model is not None:
            pred_e, pred_f = oracle_model(da, db)
        else:
            msg_a, _ = sender(da)
            msg_b, _ = sender(db)
            pred_e, pred_f = receiver(msg_a, msg_b)

        pred_e_bin = pred_e > 0
        pred_f_bin = pred_f > 0

        e_diff = torch.tensor(e_bins[ia] != e_bins[ib], device=device)
        f_diff = torch.tensor(f_bins[ia] != f_bins[ib], device=device)

        if e_diff.sum() > 0:
            correct_e += (pred_e_bin[e_diff] == label_e[e_diff]).sum().item()
            total_e += e_diff.sum().item()
        if f_diff.sum() > 0:
            correct_f += (pred_f_bin[f_diff] == label_f[f_diff]).sum().item()
            total_f += f_diff.sum().item()
        both_diff = e_diff & f_diff
        if both_diff.sum() > 0:
            both_ok = (pred_e_bin[both_diff] == label_e[both_diff]) & \
                      (pred_f_bin[both_diff] == label_f[both_diff])
            correct_both += both_ok.sum().item()
            total_both += both_diff.sum().item()

    return (correct_e / max(total_e, 1),
            correct_f / max(total_f, 1),
            correct_both / max(total_both, 1))


def evaluate_population_flat(sender, receivers, data_f16, e_bins, f_bins,
                             scene_ids, device, n_rounds=30):
    best_both = 0
    best_r = None
    for r in receivers:
        _, _, both = evaluate_accuracy_flat(
            sender, r, data_f16, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if both > best_both:
            best_both = both
            best_r = r
    return evaluate_accuracy_flat(
        sender, best_r, data_f16, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds), best_r


# ══════════════════════════════════════════════════════════════════
# Evaluation (4-agent views)
# ══════════════════════════════════════════════════════════════════

def evaluate_accuracy_multi(multi_sender, receiver, data_f16, e_bins, f_bins,
                            scene_ids, device, oracle_model=None, n_rounds=30):
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    correct_e = correct_f = correct_both = 0
    total_e = total_f = total_both = 0

    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)

        views_a = split_temporal_views(data_f16, ia)
        views_a = [v.to(device) for v in views_a]
        views_b = split_temporal_views(data_f16, ib)
        views_b = [v.to(device) for v in views_b]

        label_e = (e_dev[ia] > e_dev[ib])
        label_f = (f_dev[ia] > f_dev[ib])

        if oracle_model is not None:
            pred_e, pred_f = oracle_model(views_a, views_b)
        else:
            msg_a, _ = multi_sender(views_a)
            msg_b, _ = multi_sender(views_b)
            pred_e, pred_f = receiver(msg_a, msg_b)

        pred_e_bin = pred_e > 0
        pred_f_bin = pred_f > 0

        e_diff = torch.tensor(e_bins[ia] != e_bins[ib], device=device)
        f_diff = torch.tensor(f_bins[ia] != f_bins[ib], device=device)

        if e_diff.sum() > 0:
            correct_e += (pred_e_bin[e_diff] == label_e[e_diff]).sum().item()
            total_e += e_diff.sum().item()
        if f_diff.sum() > 0:
            correct_f += (pred_f_bin[f_diff] == label_f[f_diff]).sum().item()
            total_f += f_diff.sum().item()
        both_diff = e_diff & f_diff
        if both_diff.sum() > 0:
            both_ok = (pred_e_bin[both_diff] == label_e[both_diff]) & \
                      (pred_f_bin[both_diff] == label_f[both_diff])
            correct_both += both_ok.sum().item()
            total_both += both_diff.sum().item()

    return (correct_e / max(total_e, 1),
            correct_f / max(total_f, 1),
            correct_both / max(total_both, 1))


def evaluate_population_multi(multi_sender, receivers, data_f16, e_bins, f_bins,
                              scene_ids, device, n_rounds=30):
    best_both = 0
    best_r = None
    for r in receivers:
        _, _, both = evaluate_accuracy_multi(
            multi_sender, r, data_f16, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if both > best_both:
            best_both = both
            best_r = r
    return evaluate_accuracy_multi(
        multi_sender, best_r, data_f16, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds), best_r


# ══════════════════════════════════════════════════════════════════
# Compositionality metrics
# ══════════════════════════════════════════════════════════════════

def _mutual_information(x, y):
    x_vals, y_vals = np.unique(x), np.unique(y)
    n = len(x)
    mi = 0.0
    for xv in x_vals:
        for yv in y_vals:
            p_xy = np.sum((x == xv) & (y == yv)) / n
            p_x = np.sum(x == xv) / n
            p_y = np.sum(y == yv) / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))
    return mi


def compute_compositionality_flat(sender, data_f16, e_bins, f_bins, device):
    """For 2-agent (flat input) sender."""
    sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data_f16), BATCH_SIZE):
            batch = data_f16[i:i+BATCH_SIZE].float().to(device)
            msg, logits = sender(batch)
            tokens_batch = []
            for head_logits in logits:
                tokens_batch.append(head_logits.argmax(dim=-1).cpu().numpy())
            all_tokens.append(np.stack(tokens_batch, axis=1))

    all_tokens = np.concatenate(all_tokens, axis=0)
    return _compute_comp_from_tokens(all_tokens, e_bins, f_bins, len(data_f16))


def compute_compositionality_multi(multi_sender, data_f16, e_bins, f_bins, device):
    """For 4-agent (multi-view) sender."""
    multi_sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data_f16), BATCH_SIZE):
            end = min(i + BATCH_SIZE, len(data_f16))
            indices = np.arange(i, end)
            views = split_temporal_views(data_f16, indices)
            views = [v.to(device) for v in views]
            msg, logits = multi_sender(views)
            tokens_batch = []
            for head_logits in logits:
                tokens_batch.append(head_logits.argmax(dim=-1).cpu().numpy())
            all_tokens.append(np.stack(tokens_batch, axis=1))

    all_tokens = np.concatenate(all_tokens, axis=0)
    return _compute_comp_from_tokens(all_tokens, e_bins, f_bins, len(data_f16))


def _compute_comp_from_tokens(all_tokens, e_bins, f_bins, n_scenes):
    n_pos = all_tokens.shape[1]

    # Entropy
    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / np.log(VOCAB_SIZE)))

    # MI matrix
    attributes = np.stack([e_bins, f_bins], axis=1)
    mi_matrix = np.zeros((n_pos, 2))
    for p in range(n_pos):
        for a in range(2):
            mi_matrix[p, a] = _mutual_information(all_tokens[:, p], attributes[:, a])

    # PosDis
    if n_pos >= 2:
        pos_dis = 0.0
        for p in range(n_pos):
            sorted_mi = np.sort(mi_matrix[p])[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= n_pos
    else:
        pos_dis = 0.0

    # TopSim
    rng = np.random.RandomState(42)
    n_pairs = min(5000, n_scenes * (n_scenes - 1) // 2)
    meaning_dists, message_dists = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(n_scenes, size=2, replace=False)
        meaning_dists.append(abs(int(e_bins[i]) - int(e_bins[j])) +
                             abs(int(f_bins[i]) - int(f_bins[j])))
        message_dists.append(int((all_tokens[i] != all_tokens[j]).sum()))
    topsim, _ = stats.spearmanr(meaning_dists, message_dists)
    if np.isnan(topsim):
        topsim = 0.0

    return {
        'pos_dis': float(pos_dis),
        'topsim': float(topsim),
        'entropies': entropies,
        'mi_matrix': mi_matrix,
    }


# ══════════════════════════════════════════════════════════════════
# Training: 2-agent flat (78-probe, 78b)
# ══════════════════════════════════════════════════════════════════

def train_oracle_flat(encoder_cls, enc_kwargs, data_f16, e_bins, f_bins,
                      train_ids, device, seed):
    oracle = Oracle(encoder_cls, enc_kwargs, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(oracle.parameters(), lr=ORACLE_LR)
    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    best_acc = 0.0
    best_state = None
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    for epoch in range(ORACLE_EPOCHS):
        oracle.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da = data_f16[ia].float().to(device)
            db = data_f16[ib].float().to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            pred_e, pred_f = oracle(da, db)
            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            oracle.eval()
            with torch.no_grad():
                _, _, acc_both = evaluate_accuracy_flat(
                    None, None, data_f16, e_bins, f_bins, train_ids, device,
                    oracle_model=oracle)
            if acc_both > best_acc:
                best_acc = acc_both
                best_state = {k: v.cpu().clone()
                              for k, v in oracle.state_dict().items()}

        if epoch % 20 == 0:
            torch.mps.empty_cache()

    if best_state is not None:
        oracle.load_state_dict(best_state)
    return oracle, best_acc


def train_population_flat(sender, receivers, data_f16, e_bins, f_bins,
                          train_ids, holdout_ids, device, msg_dim, seed,
                          comm_epochs=COMM_EPOCHS):
    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_both_acc = 0.0
    best_sender_state = None
    best_receiver_states = None
    nan_count = 0
    t_start = time.time()

    for epoch in range(comm_epochs):
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            gen = epoch // RECEIVER_RESET_INTERVAL
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)
            print(f"    ** Reset ALL {len(receivers)} receivers at epoch {epoch+1} "
                  f"(gen {gen}) **", flush=True)

        sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, comm_epochs - 1)
        hard = epoch >= SOFT_WARMUP
        epoch_nan = 0

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da = data_f16[ia].float().to(device)
            db = data_f16[ib].float().to(device)
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            msg_a, logits_a = sender(da, tau=tau, hard=hard)
            msg_b, logits_b = sender(db, tau=tau, hard=hard)

            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_e, pred_f = r(msg_a, msg_b)
                r_loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                         F.binary_cross_entropy_with_logits(pred_f, label_f)
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            for logits_list in [logits_a, logits_b]:
                for logits in logits_list:
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp().clamp(min=1e-8)
                    ent = -(probs * log_probs).sum(dim=-1).mean()
                    rel_ent = ent / max_entropy
                    if rel_ent < ENTROPY_THRESHOLD:
                        loss = loss - ENTROPY_COEF * ent

            if torch.isnan(loss) or torch.isinf(loss):
                sender_opt.zero_grad()
                for opt in receiver_opts:
                    opt.zero_grad()
                nan_count += 1
                epoch_nan += 1
                continue

            sender_opt.zero_grad()
            for opt in receiver_opts:
                opt.zero_grad()
            loss.backward()

            has_nan_grad = False
            all_params = list(sender.parameters())
            for r in receivers:
                all_params.extend(list(r.parameters()))
            for p in all_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                sender_opt.zero_grad()
                for opt in receiver_opts:
                    opt.zero_grad()
                nan_count += 1
                epoch_nan += 1
                continue

            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts:
                opt.step()

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        if epoch_nan > n_batches // 2 and epoch > SOFT_WARMUP:
            print(f"    WARNING: NaN divergence at epoch {epoch+1}", flush=True)
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                (te, tf, tb), best_r = evaluate_population_flat(
                    sender, receivers, data_f16, e_bins, f_bins,
                    train_ids, device, n_rounds=20)
                (he, hf, hb), _ = evaluate_population_flat(
                    sender, receivers, data_f16, e_bins, f_bins,
                    holdout_ids, device, n_rounds=20)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (comm_epochs - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""

            print(f"    Ep {epoch+1:3d}: tau={tau:.2f}  "
                  f"train[e={te:.1%} f={tf:.1%} both={tb:.1%}]  "
                  f"holdout[e={he:.1%} f={hf:.1%} both={hb:.1%}]{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if tb > best_both_acc:
                best_both_acc = tb
                best_sender_state = {k: v.cpu().clone()
                                     for k, v in sender.state_dict().items()}
                best_receiver_states = [
                    {k: v.cpu().clone() for k, v in r.state_dict().items()}
                    for r in receivers
                ]

    if best_sender_state is not None:
        sender.load_state_dict(best_sender_state)
    if best_receiver_states is not None:
        for r, s in zip(receivers, best_receiver_states):
            r.load_state_dict(s)

    return receivers, nan_count


# ══════════════════════════════════════════════════════════════════
# Training: 4-agent multi-view (78c)
# ══════════════════════════════════════════════════════════════════

def train_oracle_multi(n_agents, encoder_cls, enc_kwargs, data_f16, e_bins, f_bins,
                       train_ids, device, seed):
    oracle = MultiAgentOracle(n_agents, encoder_cls, enc_kwargs, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(oracle.parameters(), lr=ORACLE_LR)
    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    best_acc = 0.0
    best_state = None
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    for epoch in range(ORACLE_EPOCHS):
        oracle.train()
        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            views_a = split_temporal_views(data_f16, ia)
            views_a = [v.to(device) for v in views_a]
            views_b = split_temporal_views(data_f16, ib)
            views_b = [v.to(device) for v in views_b]
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            pred_e, pred_f = oracle(views_a, views_b)
            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            oracle.eval()
            with torch.no_grad():
                _, _, acc_both = evaluate_accuracy_multi(
                    None, None, data_f16, e_bins, f_bins, train_ids, device,
                    oracle_model=oracle)
            if acc_both > best_acc:
                best_acc = acc_both
                best_state = {k: v.cpu().clone()
                              for k, v in oracle.state_dict().items()}

        if epoch % 20 == 0:
            torch.mps.empty_cache()

    if best_state is not None:
        oracle.load_state_dict(best_state)
    return oracle, best_acc


def train_population_multi(multi_sender, receivers, data_f16, e_bins, f_bins,
                           train_ids, holdout_ids, device, msg_dim, seed):
    sender_opt = torch.optim.Adam(multi_sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR)
                     for r in receivers]

    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)

    max_entropy = math.log(VOCAB_SIZE)
    n_batches = max(1, len(train_ids) // BATCH_SIZE)

    best_both_acc = 0.0
    best_sender_state = None
    best_receiver_states = None
    nan_count = 0
    t_start = time.time()

    for epoch in range(COMM_EPOCHS):
        if epoch > 0 and epoch % RECEIVER_RESET_INTERVAL == 0:
            gen = epoch // RECEIVER_RESET_INTERVAL
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(
                    receivers[i].parameters(), lr=RECEIVER_LR)
            print(f"    ** Reset ALL {len(receivers)} receivers at epoch {epoch+1} "
                  f"(gen {gen}) **", flush=True)

        multi_sender.train()
        for r in receivers:
            r.train()

        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP
        epoch_nan = 0

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            views_a = split_temporal_views(data_f16, ia)
            views_a = [v.to(device) for v in views_a]
            views_b = split_temporal_views(data_f16, ib)
            views_b = [v.to(device) for v in views_b]
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()

            msg_a, logits_a = multi_sender(views_a, tau=tau, hard=hard)
            msg_b, logits_b = multi_sender(views_b, tau=tau, hard=hard)

            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                pred_e, pred_f = r(msg_a, msg_b)
                r_loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                         F.binary_cross_entropy_with_logits(pred_f, label_f)
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            for logits_list in [logits_a, logits_b]:
                for logits in logits_list:
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp().clamp(min=1e-8)
                    ent = -(probs * log_probs).sum(dim=-1).mean()
                    rel_ent = ent / max_entropy
                    if rel_ent < ENTROPY_THRESHOLD:
                        loss = loss - ENTROPY_COEF * ent

            if torch.isnan(loss) or torch.isinf(loss):
                sender_opt.zero_grad()
                for opt in receiver_opts:
                    opt.zero_grad()
                nan_count += 1
                epoch_nan += 1
                continue

            sender_opt.zero_grad()
            for opt in receiver_opts:
                opt.zero_grad()
            loss.backward()

            has_nan_grad = False
            all_params = list(multi_sender.parameters())
            for r in receivers:
                all_params.extend(list(r.parameters()))
            for p in all_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                sender_opt.zero_grad()
                for opt in receiver_opts:
                    opt.zero_grad()
                nan_count += 1
                epoch_nan += 1
                continue

            torch.nn.utils.clip_grad_norm_(multi_sender.parameters(), 1.0)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts:
                opt.step()

        if epoch % 50 == 0:
            torch.mps.empty_cache()

        if epoch_nan > n_batches // 2 and epoch > SOFT_WARMUP:
            print(f"    WARNING: NaN divergence at epoch {epoch+1}", flush=True)
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            multi_sender.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                (te, tf, tb), best_r = evaluate_population_multi(
                    multi_sender, receivers, data_f16, e_bins, f_bins,
                    train_ids, device, n_rounds=20)
                (he, hf, hb), _ = evaluate_population_multi(
                    multi_sender, receivers, data_f16, e_bins, f_bins,
                    holdout_ids, device, n_rounds=20)

            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""

            print(f"    Ep {epoch+1:3d}: tau={tau:.2f}  "
                  f"train[e={te:.1%} f={tf:.1%} both={tb:.1%}]  "
                  f"holdout[e={he:.1%} f={hf:.1%} both={hb:.1%}]{nan_str}  "
                  f"ETA {eta/60:.0f}min", flush=True)

            if tb > best_both_acc:
                best_both_acc = tb
                best_sender_state = {k: v.cpu().clone()
                                     for k, v in multi_sender.state_dict().items()}
                best_receiver_states = [
                    {k: v.cpu().clone() for k, v in r.state_dict().items()}
                    for r in receivers
                ]

    if best_sender_state is not None:
        multi_sender.load_state_dict(best_sender_state)
    if best_receiver_states is not None:
        for r, s in zip(receivers, best_receiver_states):
            r.load_state_dict(s)

    return receivers, nan_count


# ══════════════════════════════════════════════════════════════════
# Seed runners
# ══════════════════════════════════════════════════════════════════

def run_seed_flat(seed, encoder_cls, enc_kwargs, data_f16, e_bins, f_bins,
                  train_ids, holdout_ids, device, label, comm_epochs=COMM_EPOCHS):
    """Run one seed for 2-agent flat experiment (78-probe or 78b)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    print(f"\n  --- Seed {seed} ---", flush=True)

    # Oracle
    oracle, oracle_acc = train_oracle_flat(
        encoder_cls, enc_kwargs, data_f16, e_bins, f_bins, train_ids, device, seed)
    oracle_enc_state = oracle.enc_a.state_dict()
    print(f"    Oracle: {oracle_acc:.1%}", flush=True)

    # Sender
    encoder = encoder_cls(**enc_kwargs)
    encoder.load_state_dict(oracle_enc_state)
    sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
    msg_dim = VOCAB_SIZE * N_HEADS

    # Receivers
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    print(f"    Training {label} sender (2x{VOCAB_SIZE}) vs {N_RECEIVERS} receivers "
          f"(IL={RECEIVER_RESET_INTERVAL})...", flush=True)

    receivers, nan_count = train_population_flat(
        sender, receivers, data_f16, e_bins, f_bins,
        train_ids, holdout_ids, device, msg_dim, seed,
        comm_epochs=comm_epochs)

    # Final eval
    sender.eval()
    for r in receivers:
        r.eval()
    with torch.no_grad():
        (te, tf, tb), best_r = evaluate_population_flat(
            sender, receivers, data_f16, e_bins, f_bins,
            train_ids, device, n_rounds=50)
        he, hf, hb = evaluate_accuracy_flat(
            sender, best_r, data_f16, e_bins, f_bins,
            holdout_ids, device, n_rounds=50)

    with torch.no_grad():
        comp = compute_compositionality_flat(sender, data_f16, e_bins, f_bins, device)

    mi = comp['mi_matrix']
    best_mi_e = float(mi[:, 0].max())
    best_mi_f = float(mi[:, 1].max())
    dt = time.time() - t0

    print(f"    -> holdout={hb:.1%}  PosDis={comp['pos_dis']:.3f}  "
          f"MI->e={best_mi_e:.3f}  MI->f={best_mi_f:.3f}  "
          f"NaN={nan_count}  ({dt:.0f}s)", flush=True)

    torch.mps.empty_cache()

    return {
        'seed': seed,
        'oracle_both': oracle_acc,
        'train_e': te, 'train_f': tf, 'train_both': tb,
        'holdout_e': he, 'holdout_f': hf, 'holdout_both': hb,
        'pos_dis': comp['pos_dis'],
        'topsim': comp['topsim'],
        'entropies': comp['entropies'],
        'mi_matrix': mi.tolist(),
        'best_mi_e': best_mi_e,
        'best_mi_f': best_mi_f,
        'nan_count': nan_count,
        'time_sec': dt,
    }


def run_seed_multi(seed, n_agents, encoder_cls, enc_kwargs, data_f16,
                   e_bins, f_bins, train_ids, holdout_ids, device, label):
    """Run one seed for 4-agent multi-view experiment (78c)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    print(f"\n  --- Seed {seed} ---", flush=True)

    # Oracle
    oracle, oracle_acc = train_oracle_multi(
        n_agents, encoder_cls, enc_kwargs, data_f16, e_bins, f_bins,
        train_ids, device, seed)
    print(f"    Oracle: {oracle_acc:.1%}", flush=True)

    # Per-agent senders initialized from oracle encoders
    senders = []
    for i in range(n_agents):
        enc = encoder_cls(**enc_kwargs)
        enc.load_state_dict(oracle.encs_a[i].state_dict())
        s = CompositionalSender(enc, HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
        senders.append(s)
    multi_sender = MultiAgentSender(senders).to(device)
    msg_dim = n_agents * VOCAB_SIZE * N_HEADS  # 4*5*2=40

    # Receivers
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                 for _ in range(N_RECEIVERS)]

    print(f"    Training {label} {n_agents}-agent sender "
          f"({n_agents}x2x{VOCAB_SIZE}) vs {N_RECEIVERS} receivers...", flush=True)

    receivers, nan_count = train_population_multi(
        multi_sender, receivers, data_f16, e_bins, f_bins,
        train_ids, holdout_ids, device, msg_dim, seed)

    # Final eval
    multi_sender.eval()
    for r in receivers:
        r.eval()
    with torch.no_grad():
        (te, tf, tb), best_r = evaluate_population_multi(
            multi_sender, receivers, data_f16, e_bins, f_bins,
            train_ids, device, n_rounds=50)
        he, hf, hb = evaluate_accuracy_multi(
            multi_sender, best_r, data_f16, e_bins, f_bins,
            holdout_ids, device, n_rounds=50)

    with torch.no_grad():
        comp = compute_compositionality_multi(
            multi_sender, data_f16, e_bins, f_bins, device)

    mi = comp['mi_matrix']
    best_mi_e = float(mi[:, 0].max())
    best_mi_f = float(mi[:, 1].max())
    dt = time.time() - t0

    print(f"    -> holdout={hb:.1%}  PosDis={comp['pos_dis']:.3f}  "
          f"MI->e={best_mi_e:.3f}  MI->f={best_mi_f:.3f}  "
          f"NaN={nan_count}  ({dt:.0f}s)", flush=True)

    torch.mps.empty_cache()

    return {
        'seed': seed,
        'oracle_both': oracle_acc,
        'train_e': te, 'train_f': tf, 'train_both': tb,
        'holdout_e': he, 'holdout_f': hf, 'holdout_both': hb,
        'pos_dis': comp['pos_dis'],
        'topsim': comp['topsim'],
        'entropies': comp['entropies'],
        'mi_matrix': mi.tolist(),
        'best_mi_e': best_mi_e,
        'best_mi_f': best_mi_f,
        'nan_count': nan_count,
        'time_sec': dt,
    }


# ══════════════════════════════════════════════════════════════════
# Result summary helper
# ══════════════════════════════════════════════════════════════════

def summarize_results(all_results, label):
    hb = [r['holdout_both'] for r in all_results]
    pd = [r['pos_dis'] for r in all_results]
    ts = [r['topsim'] for r in all_results]
    me = [r['best_mi_e'] for r in all_results]
    mf = [r['best_mi_f'] for r in all_results]

    compositional = [r for r in all_results if r['pos_dis'] > 0.4]
    holistic = [r for r in all_results if r['pos_dis'] < 0.15]
    n = len(all_results)

    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS: {label} ({n} seeds)", flush=True)
    print(f"{'='*70}", flush=True)

    header = (f"  {'Seed':>4} | {'Holdout':>8} | {'PosDis':>7} | "
              f"{'TopSim':>7} | {'MI->e':>7} | {'MI->f':>7} | {'NaN':>3}")
    print(header, flush=True)
    print(f"  {'----':>4}-+-{'--------':>8}-+-{'-------':>7}-+-"
          f"{'-------':>7}-+-{'-------':>7}-+-{'-------':>7}-+-{'---':>3}", flush=True)

    for r in all_results:
        tag = " *" if r['pos_dis'] > 0.4 else ""
        print(f"  {r['seed']:>4} | {r['holdout_both']:>7.1%} | "
              f"{r['pos_dis']:>7.3f} | {r['topsim']:>7.3f} | "
              f"{r['best_mi_e']:>7.3f} | {r['best_mi_f']:>7.3f} | "
              f"{r['nan_count']:>3}{tag}", flush=True)

    print(f"  {'----':>4}-+-{'--------':>8}-+-{'-------':>7}-+-"
          f"{'-------':>7}-+-{'-------':>7}-+-{'-------':>7}-+-{'---':>3}", flush=True)
    print(f"  {'Mean':>4} | {np.mean(hb):>7.1%} | "
          f"{np.mean(pd):>7.3f} | {np.mean(ts):>7.3f} | "
          f"{np.mean(me):>7.3f} | {np.mean(mf):>7.3f} |", flush=True)
    print(f"  {'Std':>4} | {np.std(hb):>7.1%} | "
          f"{np.std(pd):>7.3f} | {np.std(ts):>7.3f} | "
          f"{np.std(me):>7.3f} | {np.std(mf):>7.3f} |", flush=True)

    print(f"\n  Compositional (PosDis > 0.4):  {len(compositional):>2}/{n} "
          f"({len(compositional)/n:.0%})", flush=True)
    if compositional:
        c_hb = [r['holdout_both'] for r in compositional]
        c_pd = [r['pos_dis'] for r in compositional]
        print(f"    Mean holdout: {np.mean(c_hb):.1%} +/- {np.std(c_hb):.1%}", flush=True)
        print(f"    Mean PosDis:  {np.mean(c_pd):.3f} +/- {np.std(c_pd):.3f}", flush=True)

    print(f"  Holistic (PosDis < 0.15):      {len(holistic):>2}/{n} "
          f"({len(holistic)/n:.0%})", flush=True)

    return {
        'holdout_both_mean': float(np.mean(hb)),
        'holdout_both_std': float(np.std(hb)),
        'pos_dis_mean': float(np.mean(pd)),
        'pos_dis_std': float(np.std(pd)),
        'topsim_mean': float(np.mean(ts)),
        'topsim_std': float(np.std(ts)),
        'best_mi_e_mean': float(np.mean(me)),
        'best_mi_e_std': float(np.std(me)),
        'best_mi_f_mean': float(np.mean(mf)),
        'best_mi_f_std': float(np.std(mf)),
        'compositional_count': len(compositional),
        'compositional_seeds': [r['seed'] for r in compositional],
        'holistic_count': len(holistic),
    }


# ══════════════════════════════════════════════════════════════════
# MAIN: run all three phases in sequence
# ══════════════════════════════════════════════════════════════════

def main():
    t_global = time.time()

    print("=" * 70, flush=True)
    print("V-JEPA 2 Overnight Chain: Phase 78-probe + 78b + 78c", flush=True)
    print("=" * 70, flush=True)
    print(f"  Device: {DEVICE}", flush=True)

    # Load full V-JEPA 2 token grid (stored as float16, cast per-batch)
    vjepa_path = str(RESULTS_DIR / "vjepa2_features_full.pt")
    print(f"  Loading {vjepa_path}...", flush=True)
    data_f16, e_bins, f_bins = load_vjepa_full_features(vjepa_path)
    print(f"  Features: {data_f16.shape} (dtype={data_f16.dtype})", flush=True)
    print(f"  Reshape: (300, 8, 256, 1024) — 8 temporal × 256 spatial", flush=True)

    train_ids, holdout_ids = create_splits(e_bins, f_bins, HOLDOUT_CELLS)
    print(f"  Split: {len(train_ids)} train, {len(holdout_ids)} holdout", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Phase 78-probe: Attentive probe (5 seeds, 100 epochs)
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'#'*70}", flush=True)
    print(f"# Phase 78-probe: Attentive probe (4 queries → 2048 tokens)", flush=True)
    print(f"# 5 seeds, 100 epochs (diagnostic)", flush=True)
    print(f"{'#'*70}", flush=True)

    probe_seeds = list(range(5))
    probe_enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': VJEPA_DIM, 'n_queries': 4}
    probe_results = []

    for seed in probe_seeds:
        result = run_seed_flat(
            seed, AttentiveProbeEncoder, probe_enc_kwargs,
            data_f16, e_bins, f_bins, train_ids, holdout_ids, DEVICE,
            label="probe", comm_epochs=100)
        probe_results.append(result)

        elapsed = time.time() - t_global
        avg = elapsed / len(probe_results)
        remaining_probe = avg * (len(probe_seeds) - len(probe_results))
        print(f"\n  [Probe: {len(probe_results)}/{len(probe_seeds)}, "
              f"ETA {remaining_probe/60:.0f}min]\n", flush=True)

    probe_summary = summarize_results(probe_results, "Phase 78-probe (attentive)")

    # Save probe results
    probe_save = {
        'config': {
            'encoder': 'AttentiveProbe (4 queries × 2048 tokens)',
            'n_queries': 4, 'comm_epochs': 100, 'n_seeds': 5,
        },
        'per_seed': probe_results,
        'summary': probe_summary,
    }
    with open(RESULTS_DIR / "phase78_probe.json", 'w') as f:
        json.dump(probe_save, f, indent=2)
    print(f"\n  Saved phase78_probe.json", flush=True)

    torch.mps.empty_cache()

    # ══════════════════════════════════════════════════════════════
    # Phase 78b: Factorized Perceiver (20 seeds, 400 epochs)
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'#'*70}", flush=True)
    print(f"# Phase 78b: Factorized Perceiver (spatial 8q×256 → temporal 4q×64)", flush=True)
    print(f"# 20 seeds, 400 epochs", flush=True)
    print(f"{'#'*70}", flush=True)

    t_78b = time.time()
    seeds_20 = list(range(20))
    perceiver_enc_kwargs = {
        'hidden_dim': HIDDEN_DIM, 'input_dim': VJEPA_DIM,
        'spatial_queries': 8, 'temporal_queries': 4,
    }
    results_78b = []

    for seed in seeds_20:
        result = run_seed_flat(
            seed, FactorizedPerceiverEncoder, perceiver_enc_kwargs,
            data_f16, e_bins, f_bins, train_ids, holdout_ids, DEVICE,
            label="perceiver")
        results_78b.append(result)

        elapsed_78b = time.time() - t_78b
        avg = elapsed_78b / len(results_78b)
        remaining = avg * (len(seeds_20) - len(results_78b))
        print(f"\n  [78b: {len(results_78b)}/{len(seeds_20)}, "
              f"ETA {remaining/60:.0f}min]\n", flush=True)

    summary_78b = summarize_results(results_78b, "Phase 78b (factorized Perceiver)")

    save_78b = {
        'config': {
            'encoder': 'FactorizedPerceiver (spatial 8q×256, temporal 4q×64)',
            'spatial_queries': 8, 'temporal_queries': 4,
            'comm_epochs': COMM_EPOCHS, 'n_seeds': 20,
        },
        'per_seed': results_78b,
        'summary': summary_78b,
    }
    with open(RESULTS_DIR / "phase78b_vjepa2_resampler.json", 'w') as f:
        json.dump(save_78b, f, indent=2)
    print(f"\n  Saved phase78b_vjepa2_resampler.json", flush=True)

    torch.mps.empty_cache()

    # ══════════════════════════════════════════════════════════════
    # Phase 78c: 4-agent temporal split (20 seeds, 400 epochs)
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'#'*70}", flush=True)
    print(f"# Phase 78c: 4-agent temporal split (per-agent Perceiver K=4, 2 blocks)", flush=True)
    print(f"# 20 seeds, 400 epochs", flush=True)
    print(f"{'#'*70}", flush=True)

    t_78c = time.time()
    agent_enc_kwargs = {
        'hidden_dim': HIDDEN_DIM, 'input_dim': VJEPA_DIM,
        'n_queries': 4, 'n_blocks': 2,
    }
    results_78c = []

    for seed in seeds_20:
        result = run_seed_multi(
            seed, 4, AgentPerceiverEncoder, agent_enc_kwargs,
            data_f16, e_bins, f_bins, train_ids, holdout_ids, DEVICE,
            label="4agent-temporal")
        results_78c.append(result)

        elapsed_78c = time.time() - t_78c
        avg = elapsed_78c / len(results_78c)
        remaining = avg * (len(seeds_20) - len(results_78c))
        print(f"\n  [78c: {len(results_78c)}/{len(seeds_20)}, "
              f"ETA {remaining/60:.0f}min]\n", flush=True)

    summary_78c = summarize_results(results_78c, "Phase 78c (4-agent temporal)")

    save_78c = {
        'config': {
            'encoder': 'AgentPerceiver (K=4 queries, 2 blocks, 512 tokens/agent)',
            'n_agents': 4, 'n_queries': 4, 'n_blocks': 2,
            'comm_epochs': COMM_EPOCHS, 'n_seeds': 20,
        },
        'per_seed': results_78c,
        'summary': summary_78c,
    }
    with open(RESULTS_DIR / "phase78c_vjepa2_4agent_temporal.json", 'w') as f:
        json.dump(save_78c, f, indent=2)
    print(f"\n  Saved phase78c_vjepa2_4agent_temporal.json", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Final comparison table
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}", flush=True)
    print(f"FULL COMPARISON TABLE", flush=True)
    print(f"{'='*70}", flush=True)

    # Load reference results
    with open(RESULTS_DIR / "phase54f_extended.json") as f:
        dino_2a = json.load(f)
    with open(RESULTS_DIR / "phase78_vjepa2.json") as f:
        vjepa_pool = json.load(f)

    rows = [
        ("DINOv2 2-agent",
         dino_2a['summary']['holdout_both_mean'],
         dino_2a['summary']['holdout_both_std'],
         dino_2a['summary']['pos_dis_mean'],
         dino_2a['summary']['pos_dis_std'],
         f"{dino_2a['groups']['compositional_count']}/20"),
        ("V-JEPA2 mean-pool",
         vjepa_pool['summary']['holdout_both_mean'],
         vjepa_pool['summary']['holdout_both_std'],
         vjepa_pool['summary']['pos_dis_mean'],
         vjepa_pool['summary']['pos_dis_std'],
         f"{vjepa_pool['groups']['compositional_count']}/20"),
        ("V-JEPA2 probe",
         probe_summary['holdout_both_mean'],
         probe_summary['holdout_both_std'],
         probe_summary['pos_dis_mean'],
         probe_summary['pos_dis_std'],
         f"{probe_summary['compositional_count']}/5"),
        ("V-JEPA2 Perceiver 2a",
         summary_78b['holdout_both_mean'],
         summary_78b['holdout_both_std'],
         summary_78b['pos_dis_mean'],
         summary_78b['pos_dis_std'],
         f"{summary_78b['compositional_count']}/20"),
        ("V-JEPA2 4-agent temp",
         summary_78c['holdout_both_mean'],
         summary_78c['holdout_both_std'],
         summary_78c['pos_dis_mean'],
         summary_78c['pos_dis_std'],
         f"{summary_78c['compositional_count']}/20"),
    ]

    print(f"\n  {'Condition':<25} | {'Holdout':<15} | {'PosDis':<15} | {'Comp':>5}",
          flush=True)
    print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*15}-+-{'-'*5}", flush=True)
    for name, hm, hs, pm, ps, comp in rows:
        print(f"  {name:<25} | {hm:.1%} +/- {hs:.1%}"
              f"{'':>2} | {pm:.3f} +/- {ps:.3f}"
              f"{'':>1} | {comp:>5}", flush=True)

    # Statistical comparisons: best V-JEPA 2 vs DINOv2
    print(f"\n  STATISTICAL TESTS (vs DINOv2 2-agent):", flush=True)
    dino_hb = [r['holdout_both'] for r in dino_2a['per_seed']]
    dino_pd = [r['pos_dis'] for r in dino_2a['per_seed']]

    for name, results in [("V-JEPA2 Perceiver 2a", results_78b),
                          ("V-JEPA2 4-agent temp", results_78c)]:
        v_hb = [r['holdout_both'] for r in results]
        v_pd = [r['pos_dis'] for r in results]
        t_hb, p_hb = stats.ttest_ind(v_hb, dino_hb)
        t_pd, p_pd = stats.ttest_ind(v_pd, dino_pd)
        d_hb = (np.mean(v_hb) - np.mean(dino_hb)) / np.sqrt(
            (np.std(v_hb)**2 + np.std(dino_hb)**2) / 2)
        print(f"    {name}:", flush=True)
        print(f"      Holdout: t={t_hb:.2f}, p={p_hb:.4f}, d={d_hb:.2f}", flush=True)
        print(f"      PosDis:  t={t_pd:.2f}, p={p_pd:.4f}", flush=True)

    # MI matrices for best seeds
    print(f"\n  BEST SEED MI MATRICES:", flush=True)
    for name, results in [("78b Perceiver", results_78b),
                          ("78c 4-agent", results_78c)]:
        comp_seeds = [r for r in results if r['pos_dis'] > 0.4]
        if comp_seeds:
            best = max(comp_seeds, key=lambda r: r['pos_dis'])
        else:
            best = max(results, key=lambda r: r['holdout_both'])
        mi = np.array(best['mi_matrix'])
        print(f"\n    {name} (seed {best['seed']}, PosDis={best['pos_dis']:.3f}, "
              f"holdout={best['holdout_both']:.1%}):", flush=True)
        print(f"      {'pos':>5} {'elast':>7} {'frict':>7}", flush=True)
        for p in range(mi.shape[0]):
            print(f"      {p:>5} {mi[p,0]:>7.3f} {mi[p,1]:>7.3f}", flush=True)

    dt = time.time() - t_global
    print(f"\n\n{'='*70}", flush=True)
    print(f"TOTAL TIME: {dt/3600:.1f} hours", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
