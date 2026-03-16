"""
Phase 82: Scale-matched backbone control — DINOv2 ViT-L collision.
Isolates pretraining objective vs model scale by running identical pipeline
to Phase 79 but with DINOv2 ViT-L/14 (304M params, same as V-JEPA 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from scipy import stats

# ── Config (matches Phase 79 exactly) ──────────────────────────────────
HIDDEN_DIM = 128
INPUT_DIM = 1024          # ViT-L dimension (same as V-JEPA 2)
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32
N_AGENTS = 4
N_FRAMES = 24
FRAMES_PER_AGENT = N_FRAMES // N_AGENTS  # 6

ORACLE_LR = 1e-3
SENDER_LR = 1e-3
RECEIVER_LR = 3e-3

ORACLE_EPOCHS = 200
COMM_EPOCHS = 400

TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30

ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03

RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3

HOLDOUT_CELLS = {(0, 1), (1, 3), (2, 0), (3, 4), (4, 2)}
N_SEEDS = 20

# ── Models (identical to Phase 79) ─────────────────────────────────────

class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024):
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
        x = x.permute(0, 2, 1)  # (B, T, D) -> (B, D, T)
        x = self.temporal(x).squeeze(-1)
        return self.fc(x)


class Oracle(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024):
        super().__init__()
        self.enc_a = TemporalEncoder(hidden_dim, input_dim)
        self.enc_b = TemporalEncoder(hidden_dim, input_dim)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.elast_head = nn.Linear(hidden_dim, 1)
        self.friction_head = nn.Linear(hidden_dim, 1)

    def forward(self, xa, xb):
        ha = self.enc_a(xa)
        hb = self.enc_b(xb)
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


# ── Utilities ───────────────────────────────────────────────────────────

def load_data():
    d = torch.load('results/dinov2_vitl_collision_features.pt',
                    map_location='cpu', weights_only=True)
    features = d['features'].float()  # float16 -> float32
    index = d['index']
    e_bins = np.array([s['mass_ratio_bin'] for s in index])
    f_bins = np.array([s['restitution_bin'] for s in index])
    return features, e_bins, f_bins


def create_splits(e_bins, f_bins):
    train_ids, holdout_ids = [], []
    for i in range(len(e_bins)):
        if (int(e_bins[i]), int(f_bins[i])) in HOLDOUT_CELLS:
            holdout_ids.append(i)
        else:
            train_ids.append(i)
    return np.array(train_ids), np.array(holdout_ids)


def sample_pairs(ids, batch_size, rng):
    ia = rng.choice(ids, size=batch_size, replace=True)
    ib = rng.choice(ids, size=batch_size, replace=True)
    return ia, ib


def split_views(data, n_agents, frames_per_agent):
    return [data[:, i*frames_per_agent:(i+1)*frames_per_agent, :]
            for i in range(n_agents)]


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


def evaluate_oracle(model, data_t, e_bins, f_bins, ids, device, n_rounds=20):
    model.eval()
    rng = np.random.RandomState(999)
    correct_e, correct_f, correct_both, total = 0, 0, 0, 0
    with torch.no_grad():
        for _ in range(n_rounds):
            ia, ib = sample_pairs(ids, BATCH_SIZE, rng)
            da = data_t[ia].to(device)
            db = data_t[ib].to(device)
            le = (e_bins[ia] > e_bins[ib]).astype(np.float32)
            lf = (f_bins[ia] > f_bins[ib]).astype(np.float32)
            pe, pf = model(da, db)
            pe = (pe.cpu().numpy() > 0).astype(np.float32)
            pf = (pf.cpu().numpy() > 0).astype(np.float32)
            correct_e += (pe == le).sum()
            correct_f += (pf == lf).sum()
            correct_both += ((pe == le) & (pf == lf)).sum()
            total += len(ia)
    return correct_e/total, correct_f/total, correct_both/total


def evaluate_multi_oracle(model, data_t, e_bins, f_bins, ids, device, n_rounds=20):
    model.eval()
    rng = np.random.RandomState(999)
    correct_e, correct_f, correct_both, total = 0, 0, 0, 0
    with torch.no_grad():
        for _ in range(n_rounds):
            ia, ib = sample_pairs(ids, BATCH_SIZE, rng)
            da = data_t[ia].to(device)
            db = data_t[ib].to(device)
            views_a = split_views(da, N_AGENTS, FRAMES_PER_AGENT)
            views_b = split_views(db, N_AGENTS, FRAMES_PER_AGENT)
            le = (e_bins[ia] > e_bins[ib]).astype(np.float32)
            lf = (f_bins[ia] > f_bins[ib]).astype(np.float32)
            pe, pf = model(views_a, views_b)
            pe = (pe.cpu().numpy() > 0).astype(np.float32)
            pf = (pf.cpu().numpy() > 0).astype(np.float32)
            correct_e += (pe == le).sum()
            correct_f += (pf == lf).sum()
            correct_both += ((pe == le) & (pf == lf)).sum()
            total += len(ia)
    return correct_e/total, correct_f/total, correct_both/total


def evaluate_comm_2agent(sender, receivers, data_t, e_bins, f_bins, ids, device, n_rounds=20):
    """Evaluate 2-agent communication, return best receiver results."""
    best_both = -1
    best_result = None
    for r in receivers:
        r.eval()
    sender.eval()
    for ri, r in enumerate(receivers):
        correct_e, correct_f, correct_both, total = 0, 0, 0, 0
        rng = np.random.RandomState(999)
        with torch.no_grad():
            for _ in range(n_rounds):
                ia, ib = sample_pairs(ids, BATCH_SIZE, rng)
                da = data_t[ia].to(device)
                db = data_t[ib].to(device)
                le = (e_bins[ia] > e_bins[ib]).astype(np.float32)
                lf = (f_bins[ia] > f_bins[ib]).astype(np.float32)
                msg_a, _ = sender(da)
                msg_b, _ = sender(db)
                pe, pf = r(msg_a, msg_b)
                pe = (pe.cpu().numpy() > 0).astype(np.float32)
                pf = (pf.cpu().numpy() > 0).astype(np.float32)
                correct_e += (pe == le).sum()
                correct_f += (pf == lf).sum()
                correct_both += ((pe == le) & (pf == lf)).sum()
                total += len(ia)
        acc_both = correct_both / total
        if acc_both > best_both:
            best_both = acc_both
            best_result = (correct_e/total, correct_f/total, acc_both)
    return best_result


def evaluate_comm_4agent(multi_sender, receivers, data_t, e_bins, f_bins, ids, device, n_rounds=20):
    """Evaluate 4-agent communication, return best receiver results."""
    best_both = -1
    best_result = None
    for r in receivers:
        r.eval()
    multi_sender.eval()
    for ri, r in enumerate(receivers):
        correct_e, correct_f, correct_both, total = 0, 0, 0, 0
        rng = np.random.RandomState(999)
        with torch.no_grad():
            for _ in range(n_rounds):
                ia, ib = sample_pairs(ids, BATCH_SIZE, rng)
                da = data_t[ia].to(device)
                db = data_t[ib].to(device)
                views_a = split_views(da, N_AGENTS, FRAMES_PER_AGENT)
                views_b = split_views(db, N_AGENTS, FRAMES_PER_AGENT)
                le = (e_bins[ia] > e_bins[ib]).astype(np.float32)
                lf = (f_bins[ia] > f_bins[ib]).astype(np.float32)
                msg_a, _ = multi_sender(views_a)
                msg_b, _ = multi_sender(views_b)
                pe, pf = r(msg_a, msg_b)
                pe = (pe.cpu().numpy() > 0).astype(np.float32)
                pf = (pf.cpu().numpy() > 0).astype(np.float32)
                correct_e += (pe == le).sum()
                correct_f += (pf == lf).sum()
                correct_both += ((pe == le) & (pf == lf)).sum()
                total += len(ia)
        acc_both = correct_both / total
        if acc_both > best_both:
            best_both = acc_both
            best_result = (correct_e/total, correct_f/total, acc_both)
    return best_result


def compute_compositionality_2agent(sender, data_t, e_bins, f_bins, device):
    sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data_t), BATCH_SIZE):
            batch = data_t[i:i+BATCH_SIZE].to(device)
            msg, logits = sender(batch)
            tokens_batch = []
            for head_logits in logits:
                tokens_batch.append(head_logits.argmax(dim=-1).cpu().numpy())
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

    return {
        'pos_dis': float(pos_dis),
        'topsim': float(topsim),
        'entropies': entropies,
        'mi_matrix': mi_matrix.tolist(),
    }


def compute_compositionality_4agent(multi_sender, data_t, e_bins, f_bins, device):
    multi_sender.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data_t), BATCH_SIZE):
            batch = data_t[i:i+BATCH_SIZE].to(device)
            views = split_views(batch, N_AGENTS, FRAMES_PER_AGENT)
            msg, logits = multi_sender(views)
            tokens_batch = []
            for head_logits in logits:
                tokens_batch.append(head_logits.argmax(dim=-1).cpu().numpy())
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
    for agent_idx in range(N_AGENTS):
        start = agent_idx * N_HEADS
        agent_mi = mi_matrix[start:start + N_HEADS]
        agent_pd = 0.0
        for p in range(N_HEADS):
            sorted_mi = np.sort(agent_mi[p])[::-1]
            if sorted_mi[0] > 1e-10:
                agent_pd += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        agent_pd /= N_HEADS
        per_agent_posdis.append(float(agent_pd))

    # Global PosDis
    pos_dis_global = 0.0
    for p in range(n_pos):
        sorted_mi = np.sort(mi_matrix[p])[::-1]
        if sorted_mi[0] > 1e-10:
            pos_dis_global += (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
    pos_dis_global /= n_pos

    best_agent_posdis = max(per_agent_posdis)

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

    return {
        'pos_dis_global': float(pos_dis_global),
        'best_agent_posdis': float(best_agent_posdis),
        'per_agent_posdis': per_agent_posdis,
        'topsim': float(topsim),
        'entropies': entropies,
        'mi_matrix': mi_matrix.tolist(),
    }


# ── STEP 1: Oracle probe ───────────────────────────────────────────────

def step1_oracle():
    print("=" * 70, flush=True)
    print("STEP 1: Oracle Probe — DINOv2 ViT-L collision (20 seeds, 200 epochs)", flush=True)
    print("=" * 70, flush=True)

    device = torch.device('mps')
    data, e_bins, f_bins = load_data()
    train_ids, holdout_ids = create_splits(e_bins, f_bins)
    print(f"  Data: {data.shape}, train={len(train_ids)}, holdout={len(holdout_ids)}", flush=True)

    results = []
    t0_all = time.time()

    for seed in range(N_SEEDS):
        t0 = time.time()
        print(f"\n  --- Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        model = Oracle(HIDDEN_DIM, INPUT_DIM).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=ORACLE_LR)
        data_t = data.clone()
        n_batches = max(1, len(train_ids) // BATCH_SIZE)

        for epoch in range(1, ORACLE_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            for _ in range(n_batches):
                ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
                da = data_t[ia].to(device)
                db = data_t[ib].to(device)
                le = torch.tensor((e_bins[ia] > e_bins[ib]).astype(np.float32), device=device)
                lf = torch.tensor((f_bins[ia] > f_bins[ib]).astype(np.float32), device=device)
                pe, pf = model(da, db)
                loss = F.binary_cross_entropy_with_logits(pe, le) + \
                       F.binary_cross_entropy_with_logits(pf, lf)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            if epoch % 50 == 0 or epoch == 1:
                te, tf, tb = evaluate_oracle(model, data_t, e_bins, f_bins, train_ids, device)
                he, hf, hb = evaluate_oracle(model, data_t, e_bins, f_bins, holdout_ids, device)
                print(f"    Ep {epoch:3d}: train[e={te*100:.1f}% f={tf*100:.1f}% both={tb*100:.1f}%]  "
                      f"holdout[e={he*100:.1f}% f={hf*100:.1f}% both={hb*100:.1f}%]", flush=True)

            if epoch % 100 == 0:
                torch.mps.empty_cache()

        he, hf, hb = evaluate_oracle(model, data_t, e_bins, f_bins, holdout_ids, device)
        te, tf, tb = evaluate_oracle(model, data_t, e_bins, f_bins, train_ids, device)
        elapsed = time.time() - t0
        print(f"    -> holdout={hb*100:.1f}%  train={tb*100:.1f}%  ({elapsed:.0f}s)", flush=True)

        results.append({
            'seed': seed,
            'holdout_e': float(he), 'holdout_f': float(hf), 'holdout_both': float(hb),
            'train_e': float(te), 'train_f': float(tf), 'train_both': float(tb),
        })

    holdouts = [r['holdout_both'] for r in results]
    summary = {
        'holdout_both_mean': float(np.mean(holdouts)),
        'holdout_both_std': float(np.std(holdouts)),
    }
    print(f"\n  Oracle summary:", flush=True)
    print(f"    Holdout: {summary['holdout_both_mean']*100:.1f}% ± {summary['holdout_both_std']*100:.1f}%", flush=True)

    out = {'config': {'model': 'dinov2_vitl14', 'input_dim': INPUT_DIM, 'epochs': ORACLE_EPOCHS,
                      'n_seeds': N_SEEDS}, 'per_seed': results, 'summary': summary}
    with open('results/phase82_dinov2_vitl_collision_oracle.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved results/phase82_dinov2_vitl_collision_oracle.json  ({(time.time()-t0_all)/60:.1f}min)", flush=True)
    return summary


# ── STEP 2: 4-agent communication ──────────────────────────────────────

def step2_4agent():
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: 4-Agent Communication — DINOv2 ViT-L collision (20 seeds, 400 epochs)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Agent splits: 4 agents × {FRAMES_PER_AGENT} frames each", flush=True)
    print(f"  Message: {N_AGENTS}×{N_HEADS}×{VOCAB_SIZE} = {N_AGENTS*N_HEADS*VOCAB_SIZE} msg_dim", flush=True)

    device = torch.device('mps')
    data, e_bins, f_bins = load_data()
    train_ids, holdout_ids = create_splits(e_bins, f_bins)
    msg_dim = N_AGENTS * VOCAB_SIZE * N_HEADS  # 40

    results = []
    t0_all = time.time()

    for seed in range(N_SEEDS):
        t0 = time.time()
        print(f"\n  --- Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        # Build multi-agent sender
        senders = []
        for _ in range(N_AGENTS):
            enc = TemporalEncoder(HIDDEN_DIM, INPUT_DIM)
            s = CompositionalSender(enc, HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
            senders.append(s)
        multi_sender = MultiAgentSender(senders).to(device)
        sender_opt = torch.optim.Adam(multi_sender.parameters(), lr=SENDER_LR)

        receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device) for _ in range(N_RECEIVERS)]
        receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

        data_t = data.clone()
        n_batches = max(1, len(train_ids) // BATCH_SIZE)
        nan_count = 0

        for epoch in range(1, COMM_EPOCHS + 1):
            # Receiver reset
            if epoch > 1 and (epoch - 1) % RECEIVER_RESET_INTERVAL == 0:
                for i in range(len(receivers)):
                    receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                    receiver_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

            tau = TAU_START + (TAU_END - TAU_START) * (epoch - 1) / max(1, COMM_EPOCHS - 1)
            hard = epoch >= SOFT_WARMUP

            multi_sender.train()
            for r in receivers:
                r.train()

            for _ in range(n_batches):
                ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
                da = data_t[ia].to(device)
                db = data_t[ib].to(device)
                views_a = split_views(da, N_AGENTS, FRAMES_PER_AGENT)
                views_b = split_views(db, N_AGENTS, FRAMES_PER_AGENT)
                label_e = torch.tensor((e_bins[ia] > e_bins[ib]).astype(np.float32), device=device)
                label_f = torch.tensor((f_bins[ia] > f_bins[ib]).astype(np.float32), device=device)

                msg_a, logits_a = multi_sender(views_a, tau=tau, hard=hard)
                msg_b, logits_b = multi_sender(views_b, tau=tau, hard=hard)

                # Check NaN
                if torch.isnan(msg_a).any() or torch.isnan(msg_b).any():
                    nan_count += 1
                    continue

                total_loss = torch.tensor(0.0, device=device)
                for r in receivers:
                    pe, pf = r(msg_a, msg_b)
                    r_loss = F.binary_cross_entropy_with_logits(pe, label_e) + \
                             F.binary_cross_entropy_with_logits(pf, label_f)
                    total_loss = total_loss + r_loss
                loss = total_loss / len(receivers)

                # Entropy regularization
                for logits_list in [logits_a, logits_b]:
                    for lg in logits_list:
                        probs = F.softmax(lg, dim=-1)
                        ent = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
                        if ent < ENTROPY_THRESHOLD:
                            loss = loss - ENTROPY_COEF * ent

                sender_opt.zero_grad()
                for ro in receiver_opts:
                    ro.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(multi_sender.parameters(), 1.0)
                for r in receivers:
                    torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
                sender_opt.step()
                for ro in receiver_opts:
                    ro.step()

            if epoch % 50 == 0 or epoch == 1:
                te, tf, tb = evaluate_comm_4agent(multi_sender, receivers, data_t, e_bins, f_bins, train_ids, device)
                he, hf, hb = evaluate_comm_4agent(multi_sender, receivers, data_t, e_bins, f_bins, holdout_ids, device)
                eta = (time.time() - t0) / epoch * (COMM_EPOCHS - epoch) / 60
                nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
                print(f"    Ep {epoch:3d}: tau={tau:.2f}  train[e={te*100:.1f}% f={tf*100:.1f}% both={tb*100:.1f}%]  "
                      f"holdout[e={he*100:.1f}% f={hf*100:.1f}% both={hb*100:.1f}%]{nan_str}  ETA {eta:.0f}min", flush=True)

            if epoch % 100 == 0:
                torch.mps.empty_cache()

        # Final eval
        te, tf, tb = evaluate_comm_4agent(multi_sender, receivers, data_t, e_bins, f_bins, train_ids, device)
        he, hf, hb = evaluate_comm_4agent(multi_sender, receivers, data_t, e_bins, f_bins, holdout_ids, device)
        comp = compute_compositionality_4agent(multi_sender, data_t, e_bins, f_bins, device)
        elapsed = time.time() - t0
        print(f"    -> holdout={hb*100:.1f}%  PosDis={comp['pos_dis_global']:.3f}  "
              f"best_agent={comp['best_agent_posdis']:.3f}  NaN={nan_count}  ({elapsed:.0f}s)", flush=True)

        results.append({
            'seed': seed,
            'holdout_e': float(he), 'holdout_f': float(hf), 'holdout_both': float(hb),
            'train_e': float(te), 'train_f': float(tf), 'train_both': float(tb),
            'pos_dis_global': comp['pos_dis_global'],
            'best_agent_posdis': comp['best_agent_posdis'],
            'per_agent_posdis': comp['per_agent_posdis'],
            'topsim': comp['topsim'],
            'nan_count': nan_count,
        })

    holdouts = [r['holdout_both'] for r in results]
    posdis_vals = [r['pos_dis_global'] for r in results]
    n_comp = sum(1 for r in results if r['pos_dis_global'] >= 0.4)
    summary = {
        'holdout_both_mean': float(np.mean(holdouts)),
        'holdout_both_std': float(np.std(holdouts)),
        'pos_dis_mean': float(np.mean(posdis_vals)),
        'pos_dis_std': float(np.std(posdis_vals)),
        'n_compositional': n_comp,
        'n_seeds': N_SEEDS,
    }
    print(f"\n  4-Agent summary:", flush=True)
    print(f"    Holdout: {summary['holdout_both_mean']*100:.1f}% ± {summary['holdout_both_std']*100:.1f}%", flush=True)
    print(f"    PosDis:  {summary['pos_dis_mean']:.3f} ± {summary['pos_dis_std']:.3f}", flush=True)
    print(f"    Compositional: {n_comp}/{N_SEEDS}", flush=True)

    out = {'config': {'model': 'dinov2_vitl14', 'input_dim': INPUT_DIM, 'n_agents': N_AGENTS,
                      'msg_dim': msg_dim, 'epochs': COMM_EPOCHS, 'n_seeds': N_SEEDS},
           'per_seed': results, 'summary': summary}
    with open('results/phase82_dinov2_vitl_collision_4agent.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved results/phase82_dinov2_vitl_collision_4agent.json  ({(time.time()-t0_all)/60:.1f}min)", flush=True)
    return summary


# ── STEP 3: 2-agent communication ──────────────────────────────────────

def step3_2agent():
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: 2-Agent Communication — DINOv2 ViT-L collision (20 seeds, 400 epochs)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Full 24-frame input per agent, 2×5 vocab", flush=True)

    device = torch.device('mps')
    data, e_bins, f_bins = load_data()
    train_ids, holdout_ids = create_splits(e_bins, f_bins)
    msg_dim = VOCAB_SIZE * N_HEADS  # 10

    results = []
    t0_all = time.time()

    for seed in range(N_SEEDS):
        t0 = time.time()
        print(f"\n  --- Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        enc = TemporalEncoder(HIDDEN_DIM, INPUT_DIM)
        sender = CompositionalSender(enc, HIDDEN_DIM, VOCAB_SIZE, N_HEADS).to(device)
        sender_opt = torch.optim.Adam(sender.parameters(), lr=SENDER_LR)

        receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device) for _ in range(N_RECEIVERS)]
        receiver_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]

        data_t = data.clone()
        n_batches = max(1, len(train_ids) // BATCH_SIZE)
        nan_count = 0

        for epoch in range(1, COMM_EPOCHS + 1):
            if epoch > 1 and (epoch - 1) % RECEIVER_RESET_INTERVAL == 0:
                for i in range(len(receivers)):
                    receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(device)
                    receiver_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)

            tau = TAU_START + (TAU_END - TAU_START) * (epoch - 1) / max(1, COMM_EPOCHS - 1)
            hard = epoch >= SOFT_WARMUP

            sender.train()
            for r in receivers:
                r.train()

            for _ in range(n_batches):
                ia, ib = sample_pairs(train_ids, BATCH_SIZE, rng)
                da = data_t[ia].to(device)
                db = data_t[ib].to(device)
                label_e = torch.tensor((e_bins[ia] > e_bins[ib]).astype(np.float32), device=device)
                label_f = torch.tensor((f_bins[ia] > f_bins[ib]).astype(np.float32), device=device)

                msg_a, logits_a = sender(da, tau=tau, hard=hard)
                msg_b, logits_b = sender(db, tau=tau, hard=hard)

                if torch.isnan(msg_a).any() or torch.isnan(msg_b).any():
                    nan_count += 1
                    continue

                total_loss = torch.tensor(0.0, device=device)
                for r in receivers:
                    pe, pf = r(msg_a, msg_b)
                    r_loss = F.binary_cross_entropy_with_logits(pe, label_e) + \
                             F.binary_cross_entropy_with_logits(pf, label_f)
                    total_loss = total_loss + r_loss
                loss = total_loss / len(receivers)

                for logits_list in [logits_a, logits_b]:
                    for lg in logits_list:
                        probs = F.softmax(lg, dim=-1)
                        ent = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
                        if ent < ENTROPY_THRESHOLD:
                            loss = loss - ENTROPY_COEF * ent

                sender_opt.zero_grad()
                for ro in receiver_opts:
                    ro.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
                for r in receivers:
                    torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
                sender_opt.step()
                for ro in receiver_opts:
                    ro.step()

            if epoch % 50 == 0 or epoch == 1:
                te, tf, tb = evaluate_comm_2agent(sender, receivers, data_t, e_bins, f_bins, train_ids, device)
                he, hf, hb = evaluate_comm_2agent(sender, receivers, data_t, e_bins, f_bins, holdout_ids, device)
                eta = (time.time() - t0) / epoch * (COMM_EPOCHS - epoch) / 60
                nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
                print(f"    Ep {epoch:3d}: tau={tau:.2f}  train[e={te*100:.1f}% f={tf*100:.1f}% both={tb*100:.1f}%]  "
                      f"holdout[e={he*100:.1f}% f={hf*100:.1f}% both={hb*100:.1f}%]{nan_str}  ETA {eta:.0f}min", flush=True)

            if epoch % 100 == 0:
                torch.mps.empty_cache()

        # Final eval
        te, tf, tb = evaluate_comm_2agent(sender, receivers, data_t, e_bins, f_bins, train_ids, device)
        he, hf, hb = evaluate_comm_2agent(sender, receivers, data_t, e_bins, f_bins, holdout_ids, device)
        comp = compute_compositionality_2agent(sender, data_t, e_bins, f_bins, device)
        elapsed = time.time() - t0
        print(f"    -> holdout={hb*100:.1f}%  PosDis={comp['pos_dis']:.3f}  NaN={nan_count}  ({elapsed:.0f}s)", flush=True)

        results.append({
            'seed': seed,
            'holdout_e': float(he), 'holdout_f': float(hf), 'holdout_both': float(hb),
            'train_e': float(te), 'train_f': float(tf), 'train_both': float(tb),
            'pos_dis': comp['pos_dis'],
            'topsim': comp['topsim'],
            'nan_count': nan_count,
        })

    holdouts = [r['holdout_both'] for r in results]
    posdis_vals = [r['pos_dis'] for r in results]
    n_comp = sum(1 for r in results if r['pos_dis'] >= 0.4)
    summary = {
        'holdout_both_mean': float(np.mean(holdouts)),
        'holdout_both_std': float(np.std(holdouts)),
        'pos_dis_mean': float(np.mean(posdis_vals)),
        'pos_dis_std': float(np.std(posdis_vals)),
        'n_compositional': n_comp,
        'n_seeds': N_SEEDS,
    }
    print(f"\n  2-Agent summary:", flush=True)
    print(f"    Holdout: {summary['holdout_both_mean']*100:.1f}% ± {summary['holdout_both_std']*100:.1f}%", flush=True)
    print(f"    PosDis:  {summary['pos_dis_mean']:.3f} ± {summary['pos_dis_std']:.3f}", flush=True)
    print(f"    Compositional: {n_comp}/{N_SEEDS}", flush=True)

    out = {'config': {'model': 'dinov2_vitl14', 'input_dim': INPUT_DIM, 'n_agents': 1,
                      'msg_dim': msg_dim, 'epochs': COMM_EPOCHS, 'n_seeds': N_SEEDS},
           'per_seed': results, 'summary': summary}
    with open('results/phase82_dinov2_vitl_collision_2agent.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved results/phase82_dinov2_vitl_collision_2agent.json  ({(time.time()-t0_all)/60:.1f}min)", flush=True)
    return summary


# ── STEP 4: Comparison table ───────────────────────────────────────────

def step4_comparison():
    print("\n" + "=" * 70, flush=True)
    print("STEP 4: SCALE-MATCHED COMPARISON", flush=True)
    print("=" * 70, flush=True)

    # Load all results
    with open('results/phase82_dinov2_vitl_collision_oracle.json') as f:
        oracle = json.load(f)
    with open('results/phase82_dinov2_vitl_collision_4agent.json') as f:
        four_agent = json.load(f)
    with open('results/phase82_dinov2_vitl_collision_2agent.json') as f:
        two_agent = json.load(f)

    oracle_mean = oracle['summary']['holdout_both_mean']
    fa_mean = four_agent['summary']['holdout_both_mean']
    fa_std = four_agent['summary']['holdout_both_std']
    fa_posdis = four_agent['summary']['pos_dis_mean']
    fa_comp = four_agent['summary']['n_compositional']
    ta_mean = two_agent['summary']['holdout_both_mean']
    ta_std = two_agent['summary']['holdout_both_std']
    ta_posdis = two_agent['summary']['pos_dis_mean']
    ta_comp = two_agent['summary']['n_compositional']

    print(f"\n  | {'Backbone':<16} | {'Scale':>5} | {'Pretrain':>7} | {'Oracle':>7} | {'4-Ag Hold':>10} | {'PosDis':>7} | {'Comp%':>5} | {'2-Ag Hold':>10} | {'2-Ag PD':>7} |", flush=True)
    print(f"  |{'-'*18}|{'-'*7}|{'-'*9}|{'-'*9}|{'-'*12}|{'-'*9}|{'-'*7}|{'-'*12}|{'-'*9}|", flush=True)
    print(f"  | {'DINOv2 ViT-S':<16} | {'22M':>5} | {'Image':>7} | {'78.7%':>7} | {'77.7±3.9%':>10} | {'0.904':>7} | {'20/20':>5} | {'68.2±5.5%':>10} | {'0.422':>7} |", flush=True)
    print(f"  | {'DINOv2 ViT-L':<16} | {'304M':>5} | {'Image':>7} | {oracle_mean*100:5.1f}% | {fa_mean*100:5.1f}±{fa_std*100:.1f}% | {fa_posdis:5.3f} | {fa_comp:>2}/20 | {ta_mean*100:5.1f}±{ta_std*100:.1f}% | {ta_posdis:5.3f} |", flush=True)
    print(f"  | {'V-JEPA 2 ViT-L':<16} | {'304M':>5} | {'Video':>7} | {'88.0%':>7} | {'87.4±3.1%':>10} | {'0.962':>7} | {'20/20':>5} | {'76.2±5.4%':>10} | {'0.572':>7} |", flush=True)

    # Statistical tests
    print(f"\n  --- Statistical Tests ---", flush=True)

    # We need per-seed holdout values for the reference conditions
    # Phase 79 DINOv2 ViT-S 4-agent
    vits_4ag = [0.777]  # Use mean as placeholder; actual per-seed not loaded
    vitl_4ag = [r['holdout_both'] for r in four_agent['per_seed']]
    vitl_2ag = [r['holdout_both'] for r in two_agent['per_seed']]

    # Load Phase 79 reference data if available
    try:
        with open('results/phase79_dinov2_4agent_collision.json') as f:
            ref79_4ag = json.load(f)
        vits_4ag_seeds = [r['holdout_both'] for r in ref79_4ag['per_seed']]
    except (FileNotFoundError, KeyError):
        vits_4ag_seeds = None

    try:
        with open('results/phase79b_vjepa2_4agent_collision.json') as f:
            ref79b_4ag = json.load(f)
        vjepa_4ag_seeds = [r['holdout_both'] for r in ref79b_4ag['per_seed']]
    except (FileNotFoundError, KeyError):
        vjepa_4ag_seeds = None

    # Also load 2-agent references
    try:
        with open('results/phase79_dinov2_collision.json') as f:
            ref79_2ag = json.load(f)
        vits_2ag_seeds = [r['holdout_both'] for r in ref79_2ag['per_seed']]
    except (FileNotFoundError, KeyError):
        vits_2ag_seeds = None

    try:
        with open('results/phase79b_vjepa2_collision.json') as f:
            ref79b_2ag = json.load(f)
        vjepa_2ag_seeds = [r['holdout_both'] for r in ref79b_2ag['per_seed']]
    except (FileNotFoundError, KeyError):
        vjepa_2ag_seeds = None

    # DINOv2 ViT-L vs V-JEPA 2 ViT-L (SCALE MATCHED)
    if vjepa_4ag_seeds:
        t_stat, p_val = stats.ttest_ind(vitl_4ag, vjepa_4ag_seeds)
        d = (np.mean(vitl_4ag) - np.mean(vjepa_4ag_seeds)) / np.sqrt(
            (np.std(vitl_4ag)**2 + np.std(vjepa_4ag_seeds)**2) / 2)
        print(f"\n  DINOv2 ViT-L vs V-JEPA 2 ViT-L (SCALE MATCHED):", flush=True)
        print(f"    t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f}", flush=True)
        if p_val < 0.05:
            if np.mean(vitl_4ag) < np.mean(vjepa_4ag_seeds):
                print(f"    → V-JEPA 2 significantly better (pretraining objective matters)", flush=True)
            else:
                print(f"    → DINOv2 ViT-L matches or exceeds V-JEPA 2 (scale was the driver)", flush=True)
        else:
            print(f"    → No significant difference (scale may explain the gap)", flush=True)
    else:
        print(f"\n  DINOv2 ViT-L vs V-JEPA 2 ViT-L: ref data not found, using summary stats", flush=True)
        # Approximate t-test from summary statistics
        m1, s1 = np.mean(vitl_4ag), np.std(vitl_4ag)
        m2, s2 = 0.874, 0.031  # V-JEPA 2 reference
        se = np.sqrt(s1**2/len(vitl_4ag) + s2**2/20)
        t_approx = (m1 - m2) / se if se > 0 else 0
        p_approx = 2 * (1 - stats.t.cdf(abs(t_approx), df=38))
        d = (m1 - m2) / np.sqrt((s1**2 + s2**2) / 2)
        print(f"    t≈{t_approx:.3f}, p≈{p_approx:.4f}, Cohen's d≈{d:.3f}", flush=True)

    # DINOv2 ViT-L vs DINOv2 ViT-S (effect of scale)
    if vits_4ag_seeds:
        t_stat, p_val = stats.ttest_ind(vitl_4ag, vits_4ag_seeds)
        d = (np.mean(vitl_4ag) - np.mean(vits_4ag_seeds)) / np.sqrt(
            (np.std(vitl_4ag)**2 + np.std(vits_4ag_seeds)**2) / 2)
        print(f"\n  DINOv2 ViT-L vs DINOv2 ViT-S (EFFECT OF SCALE):", flush=True)
        print(f"    t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f}", flush=True)
        if p_val < 0.05 and np.mean(vitl_4ag) > np.mean(vits_4ag_seeds):
            print(f"    → Larger model significantly better (scale helps)", flush=True)
        else:
            print(f"    → Scale alone doesn't significantly help", flush=True)
    else:
        print(f"\n  DINOv2 ViT-L vs DINOv2 ViT-S: ref data not found, using summary stats", flush=True)
        m1, s1 = np.mean(vitl_4ag), np.std(vitl_4ag)
        m2, s2 = 0.777, 0.039
        se = np.sqrt(s1**2/len(vitl_4ag) + s2**2/20)
        t_approx = (m1 - m2) / se if se > 0 else 0
        p_approx = 2 * (1 - stats.t.cdf(abs(t_approx), df=38))
        d = (m1 - m2) / np.sqrt((s1**2 + s2**2) / 2)
        print(f"    t≈{t_approx:.3f}, p≈{p_approx:.4f}, Cohen's d≈{d:.3f}", flush=True)

    # Interpretation
    print(f"\n  --- Interpretation ---", flush=True)
    print(f"  If DINOv2 ViT-L << V-JEPA 2 ViT-L: pretraining objective (video vs image) matters", flush=True)
    print(f"  If DINOv2 ViT-L ≈ V-JEPA 2 ViT-L: model scale was the primary driver", flush=True)
    print(f"  If DINOv2 ViT-L >> DINOv2 ViT-S: scale helps regardless of objective", flush=True)

    # PosDis comparison
    vitl_posdis = [r['pos_dis_global'] for r in four_agent['per_seed']]
    print(f"\n  --- PosDis comparison ---", flush=True)
    print(f"  DINOv2 ViT-S 4-agent: 0.904 ± 0.063", flush=True)
    print(f"  DINOv2 ViT-L 4-agent: {np.mean(vitl_posdis):.3f} ± {np.std(vitl_posdis):.3f}", flush=True)
    print(f"  V-JEPA 2 ViT-L 4-agent: 0.962 ± 0.028", flush=True)


# ── Main ────────────────────────────────────────────────────────────────

def run_phase82():
    print("Phase 82: Scale-Matched Backbone Control — DINOv2 ViT-L Collision", flush=True)
    device = torch.device('mps')
    print(f"Device: {device}", flush=True)

    # Load and verify data
    data, e_bins, f_bins = load_data()
    print(f"DINOv2 ViT-L collision features: {data.shape} (dtype={data.dtype})", flush=True)
    print(f"Labels: mass_ratio bins {np.unique(e_bins)}, restitution bins {np.unique(f_bins)}", flush=True)

    step1_oracle()
    step2_4agent()
    step3_2agent()
    step4_comparison()

    print("\n\nPhase 82 complete.", flush=True)


if __name__ == '__main__':
    run_phase82()
