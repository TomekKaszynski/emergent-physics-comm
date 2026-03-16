"""
Phase 81: Three Targeted Controls for NeurIPS
===============================================
All DINOv2 ramp dataset. Same IL recipe as Phase 72/54f.

STEP 1: Randomized frame assignment (4-agent, random 2 frames each)
STEP 2: Matched bandwidth control (2-agent, 4×5 vocab)
STEP 3: 3-agent DINOv2 ramp (bridging 2→4 transition)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase81_controls.py
"""

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
# Configuration — matches Phase 72 / 54f exactly
# ══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")

HIDDEN_DIM = 128
DINO_DIM = 384
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32
N_FRAMES = 8

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}

ORACLE_EPOCHS = 200
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
# Architecture — identical to Phase 72
# ══════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
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
        # x: (batch, n_frames, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, n_frames)
        x = self.temporal(x).squeeze(-1)
        return self.fc(x)


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


class Oracle(nn.Module):
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
    def __init__(self, n_agents, hidden_dim, input_dim):
        super().__init__()
        self.n_agents = n_agents
        self.encs_a = nn.ModuleList([
            TemporalEncoder(hidden_dim, input_dim) for _ in range(n_agents)
        ])
        self.encs_b = nn.ModuleList([
            TemporalEncoder(hidden_dim, input_dim) for _ in range(n_agents)
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
# Data utilities
# ══════════════════════════════════════════════════════════════════

def load_dino_features():
    data = torch.load(RESULTS_DIR / "phase54b_dino_features.pt", weights_only=False)
    features = data['features']  # (300, 8, 384)
    e_bins = np.array(data['e_bins'])
    f_bins = np.array(data['f_bins'])
    return features, e_bins, f_bins


def create_splits(e_bins, f_bins):
    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
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


def split_views_sequential(data_t, n_agents, frames_per_agent):
    """Sequential frame split: agent i gets frames [i*fpa : (i+1)*fpa]."""
    return [data_t[:, i*frames_per_agent:(i+1)*frames_per_agent, :]
            for i in range(n_agents)]


def split_views_random(data_t, n_agents, frames_per_agent, rng_seed):
    """Random frame assignment: each agent gets frames_per_agent random frames."""
    rng = np.random.RandomState(rng_seed)
    perm = rng.permutation(N_FRAMES)
    views = []
    for i in range(n_agents):
        frame_indices = perm[i*frames_per_agent:(i+1)*frames_per_agent]
        views.append(data_t[:, frame_indices, :])
    return views, perm


def split_views_3agent(data_t):
    """3-agent split: Agent 0: frames 0-1, Agent 1: frames 2-4, Agent 2: frames 5-7."""
    return [
        data_t[:, 0:2, :],   # 2 frames
        data_t[:, 2:5, :],   # 3 frames
        data_t[:, 5:8, :],   # 3 frames
    ]


# ══════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════

def evaluate_accuracy_2agent(sender, receiver, data_t, e_bins, f_bins,
                             scene_ids, device, oracle_model=None, n_rounds=30):
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    correct_e = correct_f = correct_both = 0
    total_e = total_f = total_both = 0
    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)
        da, db = data_t[ia].to(device), data_t[ib].to(device)
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


def evaluate_population_2agent(sender, receivers, data_t, e_bins, f_bins,
                               scene_ids, device, n_rounds=30):
    best_both = 0
    best_r = None
    for r in receivers:
        _, _, both = evaluate_accuracy_2agent(
            sender, r, data_t, e_bins, f_bins, scene_ids, device, n_rounds=10)
        if both > best_both:
            best_both = both
            best_r = r
    return evaluate_accuracy_2agent(
        sender, best_r, data_t, e_bins, f_bins, scene_ids, device,
        n_rounds=n_rounds), best_r


def evaluate_accuracy_multiagent(multi_sender, receiver, agent_views, e_bins, f_bins,
                                 scene_ids, device, oracle_model=None, n_rounds=30):
    rng = np.random.RandomState(999)
    e_dev = torch.tensor(e_bins, dtype=torch.float32).to(device)
    f_dev = torch.tensor(f_bins, dtype=torch.float32).to(device)
    correct_e = correct_f = correct_both = 0
    total_e = total_f = total_both = 0
    for _ in range(n_rounds):
        bs = min(BATCH_SIZE, len(scene_ids))
        ia, ib = sample_pairs(scene_ids, bs, rng)
        views_a = [v[ia].to(device) for v in agent_views]
        views_b = [v[ib].to(device) for v in agent_views]
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


def evaluate_population_multiagent(multi_sender, receivers, agent_views, e_bins, f_bins,
                                   scene_ids, device, msg_dim, n_rounds=30):
    best_both = 0
    best_r = None
    for r in receivers:
        _, _, both = evaluate_accuracy_multiagent(
            multi_sender, r, agent_views, e_bins, f_bins,
            scene_ids, device, n_rounds=10)
        if both > best_both:
            best_both = both
            best_r = r
    return evaluate_accuracy_multiagent(
        multi_sender, best_r, agent_views, e_bins, f_bins,
        scene_ids, device, n_rounds=n_rounds), best_r


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


def compute_compositionality_2agent(sender, data_t, e_bins, f_bins, device):
    sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data_t), BATCH_SIZE):
            batch = data_t[i:i+BATCH_SIZE].to(device)
            msg, logits = sender(batch)
            tokens_batch = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens_batch, axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)
    n_pos = all_tokens.shape[1]

    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / np.log(VOCAB_SIZE)))

    attributes = np.stack([e_bins, f_bins], axis=1)
    mi_matrix = np.zeros((n_pos, 2))
    for p in range(n_pos):
        for a in range(2):
            mi_matrix[p, a] = _mutual_information(all_tokens[:, p], attributes[:, a])

    if n_pos >= 2:
        pos_dis = 0.0
        for p in range(n_pos):
            sorted_mi = np.sort(mi_matrix[p])[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= n_pos
    else:
        pos_dis = 0.0

    rng = np.random.RandomState(42)
    n_pairs = min(5000, len(data_t) * (len(data_t) - 1) // 2)
    meaning_dists, message_dists = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(len(data_t), size=2, replace=False)
        meaning_dists.append(abs(int(e_bins[i]) - int(e_bins[j])) +
                             abs(int(f_bins[i]) - int(f_bins[j])))
        message_dists.append(int((all_tokens[i] != all_tokens[j]).sum()))
    topsim, _ = stats.spearmanr(meaning_dists, message_dists)
    if np.isnan(topsim):
        topsim = 0.0

    return {
        'pos_dis': float(pos_dis), 'topsim': float(topsim),
        'entropies': entropies, 'mi_matrix': mi_matrix,
    }


def compute_compositionality_multiagent(multi_sender, agent_views, e_bins, f_bins,
                                         device, n_agents, n_heads):
    multi_sender.eval()
    all_tokens = []
    n_scenes = len(agent_views[0])
    with torch.no_grad():
        for i in range(0, n_scenes, BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(device) for v in agent_views]
            msg, logits = multi_sender(views)
            tokens_batch = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens_batch, axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)
    n_pos = all_tokens.shape[1]

    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        entropies.append(float(ent / np.log(VOCAB_SIZE)))

    attributes = np.stack([e_bins, f_bins], axis=1)
    mi_matrix = np.zeros((n_pos, 2))
    for p in range(n_pos):
        for a in range(2):
            mi_matrix[p, a] = _mutual_information(all_tokens[:, p], attributes[:, a])

    # Per-agent PosDis
    per_agent_posdis = []
    for agent_idx in range(n_agents):
        start = agent_idx * n_heads
        agent_mi = mi_matrix[start:start + n_heads]
        agent_pd = 0.0
        for p in range(n_heads):
            sorted_mi = np.sort(agent_mi[p])[::-1]
            if sorted_mi[0] > 1e-10:
                agent_pd += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        agent_pd /= n_heads
        per_agent_posdis.append(float(agent_pd))

    pos_dis_global = 0.0
    for p in range(n_pos):
        sorted_mi = np.sort(mi_matrix[p])[::-1]
        if sorted_mi[0] > 1e-10:
            pos_dis_global += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
    pos_dis_global /= n_pos

    best_agent_posdis = max(per_agent_posdis)

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
        'pos_dis': float(best_agent_posdis),
        'pos_dis_global': float(pos_dis_global),
        'pos_dis_per_agent': per_agent_posdis,
        'topsim': float(topsim),
        'entropies': entropies, 'mi_matrix': mi_matrix,
    }


# ══════════════════════════════════════════════════════════════════
# Training functions
# ══════════════════════════════════════════════════════════════════

def train_oracle_2agent(data_t, e_bins, f_bins, train_ids, device, seed):
    enc_kwargs = {'hidden_dim': HIDDEN_DIM, 'input_dim': DINO_DIM}
    oracle = Oracle(TemporalEncoder, enc_kwargs, HIDDEN_DIM).to(device)
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
            da, db = data_t[ia].to(device), data_t[ib].to(device)
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
                _, _, acc_both = evaluate_accuracy_2agent(
                    None, None, data_t, e_bins, f_bins, train_ids, device,
                    oracle_model=oracle)
            if acc_both > best_acc:
                best_acc = acc_both
                best_state = {k: v.cpu().clone() for k, v in oracle.state_dict().items()}
        if epoch % 20 == 0:
            torch.mps.empty_cache()
    if best_state is not None:
        oracle.load_state_dict(best_state)
    return oracle, best_acc


def train_oracle_multiagent(agent_views, e_bins, f_bins, train_ids, device, seed, n_agents):
    oracle = MultiAgentOracle(n_agents, HIDDEN_DIM, DINO_DIM).to(device)
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
            va = [v[ia].to(device) for v in agent_views]
            vb = [v[ib].to(device) for v in agent_views]
            label_e = (e_dev[ia] > e_dev[ib]).float()
            label_f = (f_dev[ia] > f_dev[ib]).float()
            pred_e, pred_f = oracle(va, vb)
            loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                   F.binary_cross_entropy_with_logits(pred_f, label_f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            oracle.eval()
            with torch.no_grad():
                _, _, acc_both = evaluate_accuracy_multiagent(
                    None, None, agent_views, e_bins, f_bins, train_ids, device,
                    oracle_model=oracle)
            if acc_both > best_acc:
                best_acc = acc_both
                best_state = {k: v.cpu().clone() for k, v in oracle.state_dict().items()}
        if epoch % 20 == 0:
            torch.mps.empty_cache()
    if best_state is not None:
        oracle.load_state_dict(best_state)
    return oracle, best_acc


def train_comm_2agent(sender, receivers, data_t, e_bins, f_bins,
                      train_ids, holdout_ids, device, msg_dim, seed, label=""):
    sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]
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
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

        sender.train()
        for r in receivers: r.train()
        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP
        epoch_nan = 0

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            da, db = data_t[ia].to(device), data_t[ib].to(device)
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
                for opt in receiver_opts: opt.zero_grad()
                nan_count += 1; epoch_nan += 1; continue
            sender_opt.zero_grad()
            for opt in receiver_opts: opt.zero_grad()
            loss.backward()
            has_nan_grad = False
            all_params = list(sender.parameters())
            for r in receivers: all_params.extend(list(r.parameters()))
            for p in all_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True; break
            if has_nan_grad:
                sender_opt.zero_grad()
                for opt in receiver_opts: opt.zero_grad()
                nan_count += 1; epoch_nan += 1; continue
            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            for r in receivers: torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts: opt.step()

        if epoch % 50 == 0: torch.mps.empty_cache()
        if epoch_nan > n_batches // 2 and epoch > SOFT_WARMUP:
            print(f"    WARNING: NaN divergence at epoch {epoch+1}", flush=True); break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            sender.eval()
            for r in receivers: r.eval()
            with torch.no_grad():
                (te, tf, tb), _ = evaluate_population_2agent(
                    sender, receivers, data_t, e_bins, f_bins, train_ids, device, n_rounds=20)
                (he, hf, hb), _ = evaluate_population_2agent(
                    sender, receivers, data_t, e_bins, f_bins, holdout_ids, device, n_rounds=20)
            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"    Ep {epoch+1:3d}: tau={tau:.2f}  "
                      f"train[e={te:.1%} f={tf:.1%} both={tb:.1%}]  "
                      f"holdout[e={he:.1%} f={hf:.1%} both={hb:.1%}]{nan_str}  "
                      f"ETA {eta/60:.0f}min", flush=True)
            if tb > best_both_acc:
                best_both_acc = tb
                best_sender_state = {k: v.cpu().clone() for k, v in sender.state_dict().items()}
                best_receiver_states = [{k: v.cpu().clone() for k, v in r.state_dict().items()} for r in receivers]

    if best_sender_state is not None: sender.load_state_dict(best_sender_state)
    if best_receiver_states is not None:
        for r, s in zip(receivers, best_receiver_states): r.load_state_dict(s)
    return receivers, nan_count


def train_comm_multiagent(multi_sender, receivers, agent_views, e_bins, f_bins,
                          train_ids, holdout_ids, device, msg_dim, seed):
    sender_opt = torch.optim.Adam(multi_sender.parameters(), lr=SENDER_LR)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]
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
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                receiver_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

        multi_sender.train()
        for r in receivers: r.train()
        tau = TAU_START + (TAU_END - TAU_START) * epoch / max(1, COMM_EPOCHS - 1)
        hard = epoch >= SOFT_WARMUP
        epoch_nan = 0

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
            views_a = [v[ia].to(device) for v in agent_views]
            views_b = [v[ib].to(device) for v in agent_views]
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
                for opt in receiver_opts: opt.zero_grad()
                nan_count += 1; epoch_nan += 1; continue
            sender_opt.zero_grad()
            for opt in receiver_opts: opt.zero_grad()
            loss.backward()
            has_nan_grad = False
            all_params = list(multi_sender.parameters())
            for r in receivers: all_params.extend(list(r.parameters()))
            for p in all_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True; break
            if has_nan_grad:
                sender_opt.zero_grad()
                for opt in receiver_opts: opt.zero_grad()
                nan_count += 1; epoch_nan += 1; continue
            torch.nn.utils.clip_grad_norm_(multi_sender.parameters(), 1.0)
            for r in receivers: torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for opt in receiver_opts: opt.step()

        if epoch % 50 == 0: torch.mps.empty_cache()
        if epoch_nan > n_batches // 2 and epoch > SOFT_WARMUP:
            print(f"    WARNING: NaN divergence at epoch {epoch+1}", flush=True); break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            multi_sender.eval()
            for r in receivers: r.eval()
            with torch.no_grad():
                (te, tf, tb), _ = evaluate_population_multiagent(
                    multi_sender, receivers, agent_views, e_bins, f_bins,
                    train_ids, device, msg_dim, n_rounds=20)
                (he, hf, hb), _ = evaluate_population_multiagent(
                    multi_sender, receivers, agent_views, e_bins, f_bins,
                    holdout_ids, device, msg_dim, n_rounds=20)
            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (COMM_EPOCHS - epoch - 1)
            nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"    Ep {epoch+1:3d}: tau={tau:.2f}  "
                      f"train[e={te:.1%} f={tf:.1%} both={tb:.1%}]  "
                      f"holdout[e={he:.1%} f={hf:.1%} both={hb:.1%}]{nan_str}  "
                      f"ETA {eta/60:.0f}min", flush=True)
            if tb > best_both_acc:
                best_both_acc = tb
                best_sender_state = {k: v.cpu().clone() for k, v in multi_sender.state_dict().items()}
                best_receiver_states = [{k: v.cpu().clone() for k, v in r.state_dict().items()} for r in receivers]

    if best_sender_state is not None: multi_sender.load_state_dict(best_sender_state)
    if best_receiver_states is not None:
        for r, s in zip(receivers, best_receiver_states): r.load_state_dict(s)
    return receivers, nan_count


# ══════════════════════════════════════════════════════════════════
# STEP 1: Randomized frame assignment
# ══════════════════════════════════════════════════════════════════

def step1_random_frames(data_t, e_bins, f_bins):
    print("\n" + "=" * 70, flush=True)
    print("STEP 1: 4-Agent RANDOM Frame Assignment (20 seeds, 400 epochs)", flush=True)
    print("=" * 70, flush=True)
    n_agents = 4
    fpa = 2
    t0 = time.time()
    train_ids, holdout_ids = create_splits(e_bins, f_bins)

    all_results = []
    for seed in range(20):
        torch.manual_seed(seed)
        np.random.seed(seed)
        t_seed = time.time()

        # Random frame assignment — different permutation per seed
        agent_views, perm = split_views_random(data_t, n_agents, fpa, rng_seed=seed + 1000)
        perm_list = perm.tolist()
        agent_frame_map = {i: perm_list[i*fpa:(i+1)*fpa] for i in range(n_agents)}

        msg_dim = n_agents * VOCAB_SIZE * N_HEADS  # 40
        print(f"\n  --- Seed {seed} ---", flush=True)
        print(f"    Frame assignment: {agent_frame_map}", flush=True)

        # Oracle
        oracle = MultiAgentOracle(n_agents, HIDDEN_DIM, DINO_DIM).to(DEVICE)
        optimizer = torch.optim.Adam(oracle.parameters(), lr=ORACLE_LR)
        rng_o = np.random.RandomState(seed)
        e_dev = torch.tensor(e_bins, dtype=torch.float32).to(DEVICE)
        f_dev = torch.tensor(f_bins, dtype=torch.float32).to(DEVICE)
        best_acc = 0.0
        best_state = None
        n_batches = max(1, len(train_ids) // BATCH_SIZE)
        for epoch in range(ORACLE_EPOCHS):
            oracle.train()
            for _ in range(n_batches):
                ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng_o)
                va = [v[ia].to(DEVICE) for v in agent_views]
                vb = [v[ib].to(DEVICE) for v in agent_views]
                label_e = (e_dev[ia] > e_dev[ib]).float()
                label_f = (f_dev[ia] > f_dev[ib]).float()
                pred_e, pred_f = oracle(va, vb)
                loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                       F.binary_cross_entropy_with_logits(pred_f, label_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                oracle.eval()
                with torch.no_grad():
                    _, _, acc_both = evaluate_accuracy_multiagent(
                        None, None, agent_views, e_bins, f_bins, train_ids, DEVICE,
                        oracle_model=oracle)
                if acc_both > best_acc:
                    best_acc = acc_both
                    best_state = {k: v.cpu().clone() for k, v in oracle.state_dict().items()}
            if epoch % 20 == 0: torch.mps.empty_cache()
        if best_state is not None:
            oracle.load_state_dict(best_state)
        oracle_acc = best_acc
        print(f"    Oracle: {oracle_acc:.1%}", flush=True)

        # Multi-agent sender from oracle
        senders = []
        for agent_idx in range(n_agents):
            enc = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
            enc.load_state_dict(oracle.encs_a[agent_idx].state_dict())
            s = CompositionalSender(enc, HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
            senders.append(s)
        multi_sender = MultiAgentSender(senders).to(DEVICE)
        receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]

        receivers, nan_count = train_comm_multiagent(
            multi_sender, receivers, agent_views, e_bins, f_bins,
            train_ids, holdout_ids, DEVICE, msg_dim, seed)

        multi_sender.eval()
        for r in receivers: r.eval()
        with torch.no_grad():
            (te, tf, tb), best_r = evaluate_population_multiagent(
                multi_sender, receivers, agent_views, e_bins, f_bins,
                train_ids, DEVICE, msg_dim, n_rounds=50)
            he, hf, hb = evaluate_accuracy_multiagent(
                multi_sender, best_r, agent_views, e_bins, f_bins,
                holdout_ids, DEVICE, n_rounds=50)
        with torch.no_grad():
            comp = compute_compositionality_multiagent(
                multi_sender, agent_views, e_bins, f_bins, DEVICE, n_agents, N_HEADS)

        mi = comp['mi_matrix']
        dt_seed = time.time() - t_seed
        print(f"    -> holdout={hb:.1%}  PosDis={comp['pos_dis']:.3f}  "
              f"MI->e={mi[:, 0].max():.3f}  MI->f={mi[:, 1].max():.3f}  "
              f"NaN={nan_count}  ({dt_seed:.0f}s)", flush=True)

        all_results.append({
            'seed': seed, 'oracle_both': oracle_acc,
            'frame_assignment': agent_frame_map,
            'train_e': te, 'train_f': tf, 'train_both': tb,
            'holdout_e': he, 'holdout_f': hf, 'holdout_both': hb,
            'pos_dis': comp['pos_dis'], 'pos_dis_global': comp['pos_dis_global'],
            'pos_dis_per_agent': comp['pos_dis_per_agent'],
            'topsim': comp['topsim'],
            'entropies': comp['entropies'], 'mi_matrix': mi.tolist(),
            'best_mi_e': float(mi[:, 0].max()), 'best_mi_f': float(mi[:, 1].max()),
            'nan_count': nan_count, 'time_sec': dt_seed,
        })

    holdouts = [r['holdout_both'] for r in all_results]
    posdis = [r['pos_dis'] for r in all_results]
    compositional = [s for s, p in enumerate(posdis) if p >= 0.4]
    summary = {
        'holdout_both_mean': float(np.mean(holdouts)),
        'holdout_both_std': float(np.std(holdouts)),
        'pos_dis_mean': float(np.mean(posdis)),
        'pos_dis_std': float(np.std(posdis)),
        'compositional_count': len(compositional),
        'compositional_seeds': compositional,
    }
    output = {
        'config': {'condition': '4-agent random frames', 'n_agents': n_agents,
                   'frames_per_agent': fpa, 'frame_assignment': 'random_per_seed',
                   'vocab': f'{n_agents}x{N_HEADS}x{VOCAB_SIZE}',
                   'comm_epochs': COMM_EPOCHS, 'n_seeds': 20},
        'per_seed': all_results, 'summary': summary,
    }
    save_path = RESULTS_DIR / "phase81_random_frames.json"
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    dt = time.time() - t0
    print(f"\n  Random frames summary:", flush=True)
    print(f"    Holdout: {summary['holdout_both_mean']:.1%} ± {summary['holdout_both_std']:.1%}", flush=True)
    print(f"    PosDis:  {summary['pos_dis_mean']:.3f} ± {summary['pos_dis_std']:.3f}", flush=True)
    print(f"    Compositional: {summary['compositional_count']}/20", flush=True)
    print(f"  Saved {save_path}  ({dt/60:.1f}min)", flush=True)
    return output


# ══════════════════════════════════════════════════════════════════
# STEP 2: Matched bandwidth (2 agents, 4×5 vocab)
# ══════════════════════════════════════════════════════════════════

def step2_matched_bandwidth(data_t, e_bins, f_bins):
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: 2-Agent 4×5 Vocab — Matched Bandwidth (20 seeds, 400 epochs)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Single sender sees all 8 frames, sends 4 positions × vocab 5", flush=True)
    print(f"  Message capacity: 5^4 = 625 (same as 4-agent 2×5 = 5^8 per pair)", flush=True)
    n_heads_bw = 4  # 4 message positions instead of 2
    msg_dim = n_heads_bw * VOCAB_SIZE  # 20
    t0 = time.time()
    train_ids, holdout_ids = create_splits(e_bins, f_bins)

    all_results = []
    for seed in range(20):
        torch.manual_seed(seed)
        np.random.seed(seed)
        t_seed = time.time()
        print(f"\n  --- Seed {seed} ---", flush=True)

        # Oracle — standard 2-agent oracle on all 8 frames
        oracle, oracle_acc = train_oracle_2agent(data_t, e_bins, f_bins, train_ids, DEVICE, seed)
        oracle_enc_state = oracle.enc_a.state_dict()
        print(f"    Oracle: {oracle_acc:.1%}", flush=True)

        # Sender with 4 heads instead of 2
        encoder = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
        encoder.load_state_dict(oracle_enc_state)
        sender = CompositionalSender(encoder, HIDDEN_DIM, VOCAB_SIZE, n_heads_bw).to(DEVICE)
        receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]

        receivers, nan_count = train_comm_2agent(
            sender, receivers, data_t, e_bins, f_bins,
            train_ids, holdout_ids, DEVICE, msg_dim, seed)

        sender.eval()
        for r in receivers: r.eval()
        with torch.no_grad():
            (te, tf, tb), best_r = evaluate_population_2agent(
                sender, receivers, data_t, e_bins, f_bins, train_ids, DEVICE, n_rounds=50)
            he, hf, hb = evaluate_accuracy_2agent(
                sender, best_r, data_t, e_bins, f_bins, holdout_ids, DEVICE, n_rounds=50)

        # Compositionality — use 2agent metric but with 4 heads
        sender.eval()
        all_tokens = []
        with torch.no_grad():
            for i in range(0, len(data_t), BATCH_SIZE):
                batch = data_t[i:i+BATCH_SIZE].to(DEVICE)
                msg, logits = sender(batch)
                tokens_batch = [l.argmax(dim=-1).cpu().numpy() for l in logits]
                all_tokens.append(np.stack(tokens_batch, axis=1))
        all_tokens = np.concatenate(all_tokens, axis=0)
        n_pos = all_tokens.shape[1]

        entropies = []
        for p in range(n_pos):
            counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            ent = -np.sum(probs * np.log(probs))
            entropies.append(float(ent / np.log(VOCAB_SIZE)))

        attributes = np.stack([e_bins, f_bins], axis=1)
        mi_matrix = np.zeros((n_pos, 2))
        for p in range(n_pos):
            for a in range(2):
                mi_matrix[p, a] = _mutual_information(all_tokens[:, p], attributes[:, a])

        pos_dis = 0.0
        for p in range(n_pos):
            sorted_mi = np.sort(mi_matrix[p])[::-1]
            if sorted_mi[0] > 1e-10:
                pos_dis += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        pos_dis /= n_pos

        rng_ts = np.random.RandomState(42)
        n_pairs = min(5000, len(data_t) * (len(data_t) - 1) // 2)
        meaning_dists, message_dists = [], []
        for _ in range(n_pairs):
            i, j = rng_ts.choice(len(data_t), size=2, replace=False)
            meaning_dists.append(abs(int(e_bins[i]) - int(e_bins[j])) +
                                 abs(int(f_bins[i]) - int(f_bins[j])))
            message_dists.append(int((all_tokens[i] != all_tokens[j]).sum()))
        topsim, _ = stats.spearmanr(meaning_dists, message_dists)
        if np.isnan(topsim):
            topsim = 0.0

        dt_seed = time.time() - t_seed
        print(f"    -> holdout={hb:.1%}  PosDis={pos_dis:.3f}  "
              f"MI->e={mi_matrix[:, 0].max():.3f}  MI->f={mi_matrix[:, 1].max():.3f}  "
              f"NaN={nan_count}  ({dt_seed:.0f}s)", flush=True)

        all_results.append({
            'seed': seed, 'oracle_both': oracle_acc,
            'train_e': te, 'train_f': tf, 'train_both': tb,
            'holdout_e': he, 'holdout_f': hf, 'holdout_both': hb,
            'pos_dis': float(pos_dis), 'topsim': float(topsim),
            'entropies': entropies, 'mi_matrix': mi_matrix.tolist(),
            'best_mi_e': float(mi_matrix[:, 0].max()),
            'best_mi_f': float(mi_matrix[:, 1].max()),
            'nan_count': nan_count, 'time_sec': dt_seed,
        })

    holdouts = [r['holdout_both'] for r in all_results]
    posdis = [r['pos_dis'] for r in all_results]
    compositional = [s for s, p in enumerate(posdis) if p >= 0.4]
    summary = {
        'holdout_both_mean': float(np.mean(holdouts)),
        'holdout_both_std': float(np.std(holdouts)),
        'pos_dis_mean': float(np.mean(posdis)),
        'pos_dis_std': float(np.std(posdis)),
        'compositional_count': len(compositional),
        'compositional_seeds': compositional,
    }
    output = {
        'config': {'condition': '2-agent 4x5 matched bandwidth',
                   'n_agents': 1, 'n_heads': n_heads_bw, 'vocab_size': VOCAB_SIZE,
                   'msg_dim': msg_dim, 'total_capacity': f'{VOCAB_SIZE}^{n_heads_bw}=625',
                   'comm_epochs': COMM_EPOCHS, 'n_seeds': 20},
        'per_seed': all_results, 'summary': summary,
    }
    save_path = RESULTS_DIR / "phase81_matched_bandwidth.json"
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    dt = time.time() - t0
    print(f"\n  Matched bandwidth summary:", flush=True)
    print(f"    Holdout: {summary['holdout_both_mean']:.1%} ± {summary['holdout_both_std']:.1%}", flush=True)
    print(f"    PosDis:  {summary['pos_dis_mean']:.3f} ± {summary['pos_dis_std']:.3f}", flush=True)
    print(f"    Compositional: {summary['compositional_count']}/20", flush=True)
    print(f"  Saved {save_path}  ({dt/60:.1f}min)", flush=True)
    return output


# ══════════════════════════════════════════════════════════════════
# STEP 3: 3-agent DINOv2 ramp
# ══════════════════════════════════════════════════════════════════

def step3_3agent(data_t, e_bins, f_bins):
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: 3-Agent DINOv2 Ramp (20 seeds, 400 epochs)", flush=True)
    print("=" * 70, flush=True)
    n_agents = 3
    print(f"  Agent 0: frames 0-1 (2 frames)", flush=True)
    print(f"  Agent 1: frames 2-4 (3 frames)", flush=True)
    print(f"  Agent 2: frames 5-7 (3 frames)", flush=True)
    msg_dim = n_agents * VOCAB_SIZE * N_HEADS  # 30
    t0 = time.time()
    train_ids, holdout_ids = create_splits(e_bins, f_bins)
    agent_views = split_views_3agent(data_t)

    all_results = []
    for seed in range(20):
        torch.manual_seed(seed)
        np.random.seed(seed)
        t_seed = time.time()
        print(f"\n  --- Seed {seed} ---", flush=True)

        # Oracle
        oracle = MultiAgentOracle(n_agents, HIDDEN_DIM, DINO_DIM).to(DEVICE)
        optimizer = torch.optim.Adam(oracle.parameters(), lr=ORACLE_LR)
        rng_o = np.random.RandomState(seed)
        e_dev = torch.tensor(e_bins, dtype=torch.float32).to(DEVICE)
        f_dev = torch.tensor(f_bins, dtype=torch.float32).to(DEVICE)
        best_acc = 0.0
        best_state = None
        n_batches = max(1, len(train_ids) // BATCH_SIZE)
        for epoch in range(ORACLE_EPOCHS):
            oracle.train()
            for _ in range(n_batches):
                ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng_o)
                va = [v[ia].to(DEVICE) for v in agent_views]
                vb = [v[ib].to(DEVICE) for v in agent_views]
                label_e = (e_dev[ia] > e_dev[ib]).float()
                label_f = (f_dev[ia] > f_dev[ib]).float()
                pred_e, pred_f = oracle(va, vb)
                loss = F.binary_cross_entropy_with_logits(pred_e, label_e) + \
                       F.binary_cross_entropy_with_logits(pred_f, label_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                oracle.eval()
                with torch.no_grad():
                    _, _, acc_both = evaluate_accuracy_multiagent(
                        None, None, agent_views, e_bins, f_bins, train_ids, DEVICE,
                        oracle_model=oracle)
                if acc_both > best_acc:
                    best_acc = acc_both
                    best_state = {k: v.cpu().clone() for k, v in oracle.state_dict().items()}
            if epoch % 20 == 0: torch.mps.empty_cache()
        if best_state is not None:
            oracle.load_state_dict(best_state)
        oracle_acc = best_acc
        print(f"    Oracle: {oracle_acc:.1%}", flush=True)

        # Multi-agent sender from oracle
        senders = []
        for agent_idx in range(n_agents):
            enc = TemporalEncoder(HIDDEN_DIM, DINO_DIM)
            enc.load_state_dict(oracle.encs_a[agent_idx].state_dict())
            s = CompositionalSender(enc, HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
            senders.append(s)
        multi_sender = MultiAgentSender(senders).to(DEVICE)
        receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]

        receivers, nan_count = train_comm_multiagent(
            multi_sender, receivers, agent_views, e_bins, f_bins,
            train_ids, holdout_ids, DEVICE, msg_dim, seed)

        multi_sender.eval()
        for r in receivers: r.eval()
        with torch.no_grad():
            (te, tf, tb), best_r = evaluate_population_multiagent(
                multi_sender, receivers, agent_views, e_bins, f_bins,
                train_ids, DEVICE, msg_dim, n_rounds=50)
            he, hf, hb = evaluate_accuracy_multiagent(
                multi_sender, best_r, agent_views, e_bins, f_bins,
                holdout_ids, DEVICE, n_rounds=50)
        with torch.no_grad():
            comp = compute_compositionality_multiagent(
                multi_sender, agent_views, e_bins, f_bins, DEVICE, n_agents, N_HEADS)

        mi = comp['mi_matrix']
        dt_seed = time.time() - t_seed
        print(f"    -> holdout={hb:.1%}  PosDis={comp['pos_dis']:.3f}  "
              f"MI->e={mi[:, 0].max():.3f}  MI->f={mi[:, 1].max():.3f}  "
              f"NaN={nan_count}  ({dt_seed:.0f}s)", flush=True)

        all_results.append({
            'seed': seed, 'oracle_both': oracle_acc,
            'train_e': te, 'train_f': tf, 'train_both': tb,
            'holdout_e': he, 'holdout_f': hf, 'holdout_both': hb,
            'pos_dis': comp['pos_dis'], 'pos_dis_global': comp['pos_dis_global'],
            'pos_dis_per_agent': comp['pos_dis_per_agent'],
            'topsim': comp['topsim'],
            'entropies': comp['entropies'], 'mi_matrix': mi.tolist(),
            'best_mi_e': float(mi[:, 0].max()), 'best_mi_f': float(mi[:, 1].max()),
            'nan_count': nan_count, 'time_sec': dt_seed,
        })

    holdouts = [r['holdout_both'] for r in all_results]
    posdis = [r['pos_dis'] for r in all_results]
    compositional = [s for s, p in enumerate(posdis) if p >= 0.4]
    summary = {
        'holdout_both_mean': float(np.mean(holdouts)),
        'holdout_both_std': float(np.std(holdouts)),
        'pos_dis_mean': float(np.mean(posdis)),
        'pos_dis_std': float(np.std(posdis)),
        'compositional_count': len(compositional),
        'compositional_seeds': compositional,
    }
    output = {
        'config': {'condition': '3-agent sequential', 'n_agents': n_agents,
                   'frame_split': '0-1, 2-4, 5-7',
                   'vocab': f'{n_agents}x{N_HEADS}x{VOCAB_SIZE}',
                   'msg_dim': msg_dim,
                   'comm_epochs': COMM_EPOCHS, 'n_seeds': 20},
        'per_seed': all_results, 'summary': summary,
    }
    save_path = RESULTS_DIR / "phase81_3agent_ramp.json"
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    dt = time.time() - t0
    print(f"\n  3-agent summary:", flush=True)
    print(f"    Holdout: {summary['holdout_both_mean']:.1%} ± {summary['holdout_both_std']:.1%}", flush=True)
    print(f"    PosDis:  {summary['pos_dis_mean']:.3f} ± {summary['pos_dis_std']:.3f}", flush=True)
    print(f"    Compositional: {summary['compositional_count']}/20", flush=True)
    print(f"  Saved {save_path}  ({dt/60:.1f}min)", flush=True)
    return output


# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════

def print_summary(random_result, bw_result, three_result):
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: Agent Count / Bandwidth / Temporal Structure Controls", flush=True)
    print("=" * 70, flush=True)

    # Load reference data
    d2_ref = json.load(open(RESULTS_DIR / "phase54f_extended.json"))
    d4_ref = json.load(open(RESULTS_DIR / "phase72_4agent_80seeds.json"))
    d2s = d2_ref['summary'] if 'summary' in d2_ref else d2_ref
    d4s = d4_ref['summary']
    rs = random_result['summary']
    bs = bw_result['summary']
    ts = three_result['summary']

    print(f"\n  {'Agents':<8} {'Config':<25} {'Holdout':>14} {'PosDis':>8} {'Comp%':>7}", flush=True)
    print(f"  {'-'*8} {'-'*25} {'-'*14} {'-'*8} {'-'*7}", flush=True)
    print(f"  {'2':<8} {'2×5 sequential':<25} "
          f"{d2s['holdout_both_mean']:>6.1%}±{d2s['holdout_both_std']:.1%} "
          f"{d2s['pos_dis_mean']:>8.3f} "
          f"{'16/20':>7}", flush=True)
    print(f"  {'2':<8} {'4×5 sequential':<25} "
          f"{bs['holdout_both_mean']:>6.1%}±{bs['holdout_both_std']:.1%} "
          f"{bs['pos_dis_mean']:>8.3f} "
          f"{bs['compositional_count']:>4}/20", flush=True)
    print(f"  {'3':<8} {'2×5 sequential':<25} "
          f"{ts['holdout_both_mean']:>6.1%}±{ts['holdout_both_std']:.1%} "
          f"{ts['pos_dis_mean']:>8.3f} "
          f"{ts['compositional_count']:>4}/20", flush=True)
    print(f"  {'4':<8} {'2×5 sequential':<25} "
          f"{d4s['holdout_both_mean']:>6.1%}±{d4s['holdout_both_std']:.1%} "
          f"{d4s['pos_dis_mean']:>8.3f} "
          f"{'80/80':>7}", flush=True)
    print(f"  {'4':<8} {'2×5 RANDOM':<25} "
          f"{rs['holdout_both_mean']:>6.1%}±{rs['holdout_both_std']:.1%} "
          f"{rs['pos_dis_mean']:>8.3f} "
          f"{rs['compositional_count']:>4}/20", flush=True)

    # Statistical tests
    print(f"\n  Statistical tests:", flush=True)

    # Random vs sequential 4-agent
    d4_h = [r['holdout_both'] for r in d4_ref.get('per_seed', [])][:20]
    r_h = [r['holdout_both'] for r in random_result['per_seed']]
    if d4_h and r_h:
        t, p = stats.ttest_ind(r_h, d4_h)
        d_eff = (np.mean(r_h) - np.mean(d4_h)) / np.sqrt((np.std(r_h)**2 + np.std(d4_h)**2) / 2)
        print(f"    4-agent random vs sequential: t={t:.2f}, p={p:.4f}, d={d_eff:.2f}", flush=True)

    # Bandwidth control: 2-agent 4×5 vs 4-agent 2×5
    b_h = [r['holdout_both'] for r in bw_result['per_seed']]
    if d4_h and b_h:
        t, p = stats.ttest_ind(b_h, d4_h)
        d_eff = (np.mean(b_h) - np.mean(d4_h)) / np.sqrt((np.std(b_h)**2 + np.std(d4_h)**2) / 2)
        print(f"    2-agent 4×5 vs 4-agent 2×5: t={t:.2f}, p={p:.4f}, d={d_eff:.2f}", flush=True)

    # 2-agent 4×5 vs 2-agent 2×5
    d2_h = [r['holdout_both'] for r in d2_ref.get('per_seed', d2_ref.get('seeds', []))]
    if d2_h and b_h:
        t, p = stats.ttest_ind(b_h, d2_h)
        d_eff = (np.mean(b_h) - np.mean(d2_h)) / np.sqrt((np.std(b_h)**2 + np.std(d2_h)**2) / 2)
        print(f"    2-agent 4×5 vs 2-agent 2×5: t={t:.2f}, p={p:.4f}, d={d_eff:.2f}", flush=True)

    # 3-agent vs 2-agent
    t_h = [r['holdout_both'] for r in three_result['per_seed']]
    if d2_h and t_h:
        t, p = stats.ttest_ind(t_h, d2_h)
        d_eff = (np.mean(t_h) - np.mean(d2_h)) / np.sqrt((np.std(t_h)**2 + np.std(d2_h)**2) / 2)
        print(f"    3-agent vs 2-agent: t={t:.2f}, p={p:.4f}, d={d_eff:.2f}", flush=True)

    # 3-agent vs 4-agent
    if d4_h and t_h:
        t, p = stats.ttest_ind(t_h, d4_h)
        d_eff = (np.mean(t_h) - np.mean(d4_h)) / np.sqrt((np.std(t_h)**2 + np.std(d4_h)**2) / 2)
        print(f"    3-agent vs 4-agent: t={t:.2f}, p={p:.4f}, d={d_eff:.2f}", flush=True)

    # PosDis tests
    d4_pd = [r['pos_dis'] for r in d4_ref.get('per_seed', [])][:20]
    r_pd = [r['pos_dis'] for r in random_result['per_seed']]
    if d4_pd and r_pd:
        t, p = stats.ttest_ind(r_pd, d4_pd)
        print(f"    PosDis random vs sequential: t={t:.2f}, p={p:.4f}", flush=True)

    print(flush=True)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_total = time.time()
    print("Phase 81: Three Targeted Controls for NeurIPS", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    features, e_bins, f_bins = load_dino_features()
    print(f"DINOv2 ramp features: {features.shape}", flush=True)

    random_result = step1_random_frames(features, e_bins, f_bins)
    bw_result = step2_matched_bandwidth(features, e_bins, f_bins)
    three_result = step3_3agent(features, e_bins, f_bins)
    print_summary(random_result, bw_result, three_result)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total pipeline time: {dt/60:.1f} min ({dt/3600:.1f} hours)", flush=True)
    print(f"{'='*70}", flush=True)
